#!/usr/bin/env python3
"""
Bake LoRA-merged + TS encoder + AlignLayer weights into safetensors.

Supports both TOTO and TeleEncoder via GRPO_TSLLM_CONFIG.

This one-time script:
  1. Loads TSLLMForCausalLM via from_pretrained() (HF mode)
     - Loads Qwen3-4B base weights from safetensors
     - Loads TS encoder + AlignLayer from DeepSpeed checkpoint
     - Merges LoRA adapters into the language model
  2. Saves the COMPLETE model state_dict as new safetensors

After baking, the model directory contains everything — no runtime
DS checkpoint loading or LoRA merging needed.

Usage:
    export GRPO_TSLLM_CONFIG=configs/grpo_tsllm.yaml

    # Run from teleprism/
    .venv/bin/python3 -m teleprism.rl.training.bake_checkpoint \
        --model-dir ./checkpoints/grpo/model-unbaked \
        --output-dir ./checkpoints/grpo/model-baked
"""

import argparse
import json
import os
import shutil
import sys
import yaml

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Bake all weights into safetensors")
    parser.add_argument(
        "--model-dir",
        default="./checkpoints/grpo/model-unbaked",
        help="Input HF model directory (with base Qwen3-4B safetensors)",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/grpo/model-baked",
        help="Output directory for baked safetensors",
    )
    args = parser.parse_args()

    # Load encoder config (for checkpoint path and encoder type)
    grpo_cfg_path = os.environ.get("GRPO_TSLLM_CONFIG", "configs/grpo_tsllm.yaml")
    if os.path.isfile(grpo_cfg_path):
        with open(grpo_cfg_path) as f:
            grpo_cfg = yaml.safe_load(f)
        encoder_type = grpo_cfg.get("encoder", "toto")
    else:
        grpo_cfg = {}
        encoder_type = "toto"
    os.environ.setdefault("GRPO_TSLLM_CONFIG", grpo_cfg_path)

    # Checkpoint dir/tag: prefer env vars, fall back to config
    ckpt_cfg = grpo_cfg.get("checkpoint", {})
    ckpt_dir = os.environ.get("TS_CHECKPOINT_DIR") or ckpt_cfg.get("dir")
    ckpt_tag = os.environ.get("TS_CHECKPOINT_TAG") or ckpt_cfg.get("tag")
    if not ckpt_dir or not ckpt_tag:
        print("ERROR: Set TS_CHECKPOINT_DIR/TS_CHECKPOINT_TAG env vars or checkpoint.dir/tag in GRPO config")
        sys.exit(1)
    os.environ["TS_CHECKPOINT_DIR"] = ckpt_dir
    os.environ["TS_CHECKPOINT_TAG"] = ckpt_tag

    print(f"Loading model from: {args.model_dir}")
    print(f"Encoder type: {encoder_type}")
    print(f"DS checkpoint: {ckpt_dir}/{ckpt_tag}")
    print()

    # Load model in HF mode (loads base + LoRA merge + TS encoder + align)
    from teleprism.models.ts_llm.tsllm_causal.modeling_tsllm import TSLLMForCausalLM
    from transformers import AutoTokenizer

    model = TSLLMForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Resize vocab
    COLD_START_SPECIAL_TOKENS = [
        "<|begin_of_TS|>", "<|end_of_TS|>",
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    old_vocab = len(tokenizer)
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": COLD_START_SPECIAL_TOKENS}
    )
    new_vocab = len(tokenizer)

    old_embed_shape = model.language_model.get_input_embeddings().weight.shape
    model.language_model.resize_token_embeddings(new_vocab)
    new_embed_shape = model.language_model.get_input_embeddings().weight.shape
    model.config.vocab_size = new_vocab

    print(f"Tokenizer: {old_vocab} → {new_vocab} (+{n_added} tokens)")
    print(f"embed_tokens: {old_embed_shape} → {new_embed_shape}")
    print(f"Config vocab_size updated to: {new_vocab}")
    print()

    # Restore trained embed_tokens and lm_head from DS checkpoint
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    print("Loading DS checkpoint for trained embed_tokens / lm_head ...")
    ds_sd = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag=ckpt_tag)

    DS_PREFIX = "base_model.base_model.model."

    # embed_tokens
    ds_embed = ds_sd.get(f"{DS_PREFIX}model.embed_tokens.weight")
    if ds_embed is not None:
        cur_embed = model.language_model.get_input_embeddings().weight
        assert ds_embed.shape == cur_embed.shape, (
            f"embed_tokens shape mismatch: DS {ds_embed.shape} vs model {cur_embed.shape}"
        )
        cur_embed.data.copy_(ds_embed.to(dtype=cur_embed.dtype))
        print(f"  Restored embed_tokens from DS checkpoint: {ds_embed.shape}")
    else:
        print("  WARNING: embed_tokens not found in DS checkpoint!")

    ds_lmhead = ds_sd.get(f"{DS_PREFIX}lm_head.weight")
    if ds_lmhead is not None:
        cur_lmhead = model.language_model.lm_head.weight
        assert ds_lmhead.shape == cur_lmhead.shape, (
            f"lm_head shape mismatch: DS {ds_lmhead.shape} vs model {cur_lmhead.shape}"
        )
        cur_lmhead.data.copy_(ds_lmhead.to(dtype=cur_lmhead.dtype))
        print(f"  Restored lm_head from DS checkpoint: {ds_lmhead.shape}")
    else:
        print("  WARNING: lm_head not found in DS checkpoint!")

    del ds_sd  # Free memory
    print()

    # Get full state dict
    state_dict = model.state_dict()
    print(f"Total parameters in state_dict: {len(state_dict)}")

    # Count by component
    ts_params = {k: v for k, v in state_dict.items() if k.startswith("ts_encoder.")}
    align_params = {k: v for k, v in state_dict.items() if k.startswith("align_layer.")}
    lm_params = {k: v for k, v in state_dict.items() if k.startswith("language_model.")}
    print(f"  ts_encoder:     {len(ts_params)} params")
    print(f"  align_layer:    {len(align_params)} params")
    print(f"  language_model: {len(lm_params)} params")

    # Verify totals
    total_elements = sum(v.numel() for v in state_dict.values())
    print(f"  Total elements: {total_elements:,}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save as safetensors (single file for simplicity, or split if large)
    output_path = os.path.join(args.output_dir, "model.safetensors")
    print(f"Saving baked weights to: {output_path}")

    # safetensors requires contiguous tensors on CPU.
    cpu_state_dict = {k: v.contiguous().cpu().clone() for k, v in state_dict.items()}
    save_file(cpu_state_dict, output_path)

    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"  File size: {file_size:.2f} GB")
    print()

    # Create safetensors index (for compatibility with AutoWeightsLoader)
    weight_map = {k: "model.safetensors" for k in state_dict.keys()}
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in state_dict.values())},
        "weight_map": weight_map,
    }
    index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Index written: {index_path}")

    # Copy config.json, tokenizer files, hf_tsllm.py from input dir
    for fname in os.listdir(args.model_dir):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue  # Skip old safetensors
        src_path = os.path.join(args.model_dir, fname)
        dst_path = os.path.join(args.output_dir, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {fname}")

    # Save the updated tokenizer (with all special tokens)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Saved tokenizer with {len(tokenizer)} tokens")

    # Update config.json: baked flag + correct vocab_size
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["weights_baked"] = True
    config["vocab_size"] = new_vocab
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Updated config.json: weights_baked=True, vocab_size={new_vocab}")

    # Patch chat template: remove Qwen3's conditional enable_thinking logic.
    # The SFT dataloader uses prompt up to <|im_start|>assistant\n and lets
    # the model generate <think>..reasoning..</think>\n\n{answer} itself.
    # Qwen3's default template has conditional logic that can add empty
    # <think></think> blocks. We simplify to just the assistant header.
    template_path = os.path.join(args.output_dir, "chat_template.jinja")
    if os.path.isfile(template_path):
        with open(template_path) as f:
            tpl = f.read()
        old_block = (
            "{%- if add_generation_prompt %}\n"
            "    {{- '<|im_start|>assistant\\n' }}\n"
            "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
            "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
            "    {%- endif %}\n"
            "{%- endif %}"
        )
        new_block = (
            "{%- if add_generation_prompt %}\n"
            "    {{- '<|im_start|>assistant\\n' }}\n"
            "{%- endif %}"
        )
        if old_block in tpl:
            tpl = tpl.replace(old_block, new_block)
            with open(template_path, "w") as f:
                f.write(tpl)
            print(f"  Patched chat_template: clean assistant header (model generates <think> itself)")

    print()
    print(f"Baked model directory ready: {args.output_dir}")
    print()
    print("To use with VERL GRPO training:")
    print(f'  set paths.model_path="{args.output_dir}" in configs/grpo_tsllm.yaml, then: bash scripts/run_grpo_training.sh')
    print()
    print("To use for inference:")
    print(f'  point --checkpoint_dir at "{args.output_dir}" in scripts/inference_tsllm.sh')

    # Verify by reloading
    print()
    print("Verifying baked model can be loaded...")
    # Clear the DS checkpoint env vars to ensure we don't accidentally load them
    old_dir = os.environ.pop("TS_CHECKPOINT_DIR", None)
    old_tag = os.environ.pop("TS_CHECKPOINT_TAG", None)
    try:
        model2 = TSLLMForCausalLM.from_pretrained(
            args.output_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        sd2 = model2.state_dict()
        assert len(sd2) == len(state_dict), f"Param count mismatch: {len(sd2)} vs {len(state_dict)}"

        # Check a few key params are identical (encoder-aware)
        verify_keys = ["align_layer.linear.weight",
                       "language_model.model.layers.0.self_attn.q_proj.weight"]
        if encoder_type == "toto":
            verify_keys.insert(0, "ts_encoder.scaler.mean_scaler.weight")
        elif encoder_type == "teleencoder":
            verify_keys.insert(0, "ts_encoder.input_embedding.weight")
        for key in verify_keys:
            if key in state_dict and key in sd2:
                if torch.allclose(state_dict[key].cpu(), sd2[key].cpu(), atol=1e-6):
                    print(f"{key} matches")
                else:
                    print(f"{key} MISMATCH!")
        print("Verification passed!")
    finally:
        # Restore env vars
        if old_dir:
            os.environ["TS_CHECKPOINT_DIR"] = old_dir
        if old_tag:
            os.environ["TS_CHECKPOINT_TAG"] = old_tag


if __name__ == "__main__":
    main()
