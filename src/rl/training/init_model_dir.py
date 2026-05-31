#!/usr/bin/env python3
"""
Initialize a TSLLMForCausalLM model directory WITHOUT a cold-start checkpoint.

Creates a model dir with:
  - Base Qwen weights
  - Special tokens added to the tokenizer
  - Resized embeddings
  - Randomly initialized TS encoder + align layer
  - Baked safetensors (so the model loads without needing a DS checkpoint)

This is used when `checkpoint.use_cold_start: false` in grpo_tsllm.yaml,
allowing RL training to start from scratch without an SFT checkpoint.

Usage:
    export GRPO_TSLLM_CONFIG=configs/grpo_tsllm.yaml
    python3 -m teleprism.rl.training.init_model_dir \
        --model-dir /path/to/tsllm-model-dir \
        --output-dir /path/to/tsllm-model-dir-baked
"""

import argparse
import json
import os
import shutil
import sys
import yaml

import torch
from safetensors.torch import save_file


COLD_START_SPECIAL_TOKENS = [
    "<|begin_of_TS|>", "<|end_of_TS|>",
]


def main():
    parser = argparse.ArgumentParser(
        description="Initialize TSLLM model dir with random encoder (no cold-start)"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Input HF model directory (unbaked, with base Qwen weights)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for initialized model",
    )
    args = parser.parse_args()

    grpo_cfg_path = os.environ.get("GRPO_TSLLM_CONFIG", "configs/grpo_tsllm.yaml")
    os.environ.setdefault("GRPO_TSLLM_CONFIG", grpo_cfg_path)

    if os.path.isfile(grpo_cfg_path):
        with open(grpo_cfg_path) as f:
            grpo_cfg = yaml.safe_load(f)
        encoder_type = grpo_cfg.get("encoder", "toto")
    else:
        grpo_cfg = {}
        encoder_type = "toto"

    # Clear DS checkpoint env vars so the model doesn't try to load one
    os.environ.pop("TS_CHECKPOINT_DIR", None)
    os.environ.pop("TS_CHECKPOINT_TAG", None)

    print(f"Initializing model from: {args.model_dir}")
    print(f"Encoder type: {encoder_type}")
    print(f"No cold-start checkpoint — TS encoder will be randomly initialized")
    print()

    # Load model in HF mode (will create random TS encoder since no checkpoint)
    from teleprism.models.ts_llm.tsllm_causal.modeling_tsllm import TSLLMForCausalLM
    from transformers import AutoTokenizer

    model = TSLLMForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Add special tokens and resize embeddings
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
    print()

    # Optionally load pretrained TS encoder + align layer weights
    enc_init = grpo_cfg.get("checkpoint", {}).get("encoder_init", {})
    te_path = enc_init.get("teleencoder") if enc_init else None
    al_path = enc_init.get("align_layer") if enc_init else None

    if te_path and os.path.isdir(te_path):
        from safetensors.torch import load_file as _load_sf
        te_sf = os.path.join(te_path, "model.safetensors")
        if os.path.isfile(te_sf):
            te_sd = _load_sf(te_sf)
            loaded = 0
            for name, param in model.ts_encoder.named_parameters():
                if name in te_sd:
                    t = te_sd[name].to(dtype=param.dtype)
                    if param.shape == t.shape:
                        param.data.copy_(t)
                        loaded += 1
            print(f"Loaded {loaded} pretrained teleencoder params from {te_path}")
        else:
            print(f"WARNING: {te_sf} not found, using random init")
    else:
        print("No teleencoder pretrained weights — using random initialization")

    if al_path and os.path.isdir(al_path):
        from safetensors.torch import load_file as _load_sf
        al_sf = os.path.join(al_path, "model.safetensors")
        if os.path.isfile(al_sf):
            al_sd = _load_sf(al_sf)
            loaded = 0
            for name, param in model.align_layer.named_parameters():
                if name in al_sd:
                    t = al_sd[name].to(dtype=param.dtype)
                    if param.shape == t.shape:
                        param.data.copy_(t)
                        loaded += 1
            print(f"Loaded {loaded} pretrained align_layer params from {al_path}")
        else:
            print(f"WARNING: {al_sf} not found, using random init")
    else:
        print("No align_layer pretrained weights — using random initialization")

    print()

    # Get full state dict and reshape scalars to 1D (vLLM requires ≥1D tensors)
    state_dict = model.state_dict()
    n_scalars = 0
    for k in state_dict:
        if state_dict[k].dim() == 0:
            state_dict[k] = state_dict[k].unsqueeze(0)
            n_scalars += 1
    if n_scalars:
        print(f"Reshaped {n_scalars} scalar params to 1D (vLLM/FSDP compatibility)")
    print(f"Total parameters: {len(state_dict)}")

    ts_params = {k: v for k, v in state_dict.items() if k.startswith("ts_encoder.")}
    align_params = {k: v for k, v in state_dict.items() if k.startswith("align_layer.")}
    lm_params = {k: v for k, v in state_dict.items() if k.startswith("language_model.")}
    print(f"  ts_encoder:     {len(ts_params)} params (randomly initialized)")
    print(f"  align_layer:    {len(align_params)} params (randomly initialized)")
    print(f"  language_model: {len(lm_params)} params (base Qwen weights)")

    total_elements = sum(v.numel() for v in state_dict.values())
    print(f"  Total elements: {total_elements:,}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save as safetensors
    output_path = os.path.join(args.output_dir, "model.safetensors")
    print(f"Saving to: {output_path}")
    cpu_state_dict = {k: v.contiguous().cpu().clone() for k, v in state_dict.items()}
    save_file(cpu_state_dict, output_path)

    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"  File size: {file_size:.2f} GB")
    print()

    # Create safetensors index
    weight_map = {k: "model.safetensors" for k in state_dict.keys()}
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in state_dict.values())},
        "weight_map": weight_map,
    }
    index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Copy config, tokenizer files, hf_tsllm.py from input dir
    for fname in os.listdir(args.model_dir):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        src_path = os.path.join(args.model_dir, fname)
        dst_path = os.path.join(args.output_dir, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    # Save updated tokenizer
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Saved tokenizer with {len(tokenizer)} tokens")

    # Update config.json
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["weights_baked"] = True
    config["vocab_size"] = new_vocab
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Updated config.json: weights_baked=True, vocab_size={new_vocab}")

    print()
    print(f"Model directory ready: {args.output_dir}")
    print(f"  TS encoder: randomly initialized (no SFT checkpoint)")
    print(f"  LLM: base Qwen weights")


if __name__ == "__main__":
    main()
