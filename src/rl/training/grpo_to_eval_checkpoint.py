#!/usr/bin/env python3
"""Convert a verl FSDP GRPO actor checkpoint into the consolidated
``pytorch_model.pt`` format that ``evaluate_tsllm.py`` / ``inference_tsllm.load_model``
expect, so RL weights can be evaluated with the existing eval pipeline.

The GRPO actor checkpoint stores the full model as FSDP shards
(``model_world_size_K_rank_*.pt``) with PEFT-wrapped keys. Eval instead loads a
single ``pytorch_model.pt`` of just the trainable params (ts_encoder, align_layer,
LoRA, embed_tokens) keyed for the SFT ``FullModel``. This script merges the shards
and remaps the keys.

Usage:
    python -m teleprism.rl.training.grpo_to_eval_checkpoint \
        --actor_dir   <.../global_step_N/actor> \
        --template_ckpt <an SFT pytorch_model.pt providing the target key set> \
        --output_dir  <checkpoint_dir> \
        --tag         <tag>            # eval loads <output_dir>/<tag>/pytorch_model.pt
"""
import argparse
import os

import torch


def _merge_fsdp_state_dict(actor_dir):
    """Reuse verl's DTensor-aware FSDP merger to consolidate the shards."""
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    cfg = ModelMergerConfig(
        operation="test",  # avoids needing an HF target dir; we only merge
        backend="fsdp",
        local_dir=actor_dir,
        hf_model_config_path=os.path.join(actor_dir, "huggingface"),
        trust_remote_code=True,
        target_dir=None,
    )
    merger = FSDPModelMerger(cfg)
    world_size = merger._get_world_size()
    rank0 = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank0, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    return merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)


def _rl_source_key(target_key):
    """Map an SFT FullModel param name → its key in the merged GRPO state dict."""
    if target_key.startswith("ts_encoder.") or target_key.startswith("align_layer."):
        return "base_model.model." + target_key
    llm_prefix = "base_model.base_model.model.model."
    if target_key.startswith(llm_prefix):
        return "base_model.model.language_model.model." + target_key[len(llm_prefix):]
    return None


def main():
    ap = argparse.ArgumentParser(description="Convert a GRPO FSDP checkpoint to the eval pytorch_model.pt format")
    ap.add_argument("--actor_dir", required=True, help="path to <.../global_step_N/actor>")
    ap.add_argument("--template_ckpt", required=True,
                    help="an SFT pytorch_model.pt whose keys define the trainable param set")
    ap.add_argument("--output_dir", required=True, help="eval --checkpoint_dir")
    ap.add_argument("--tag", required=True,
                    help="eval --tag; output written to <output_dir>/<tag>/pytorch_model.pt")
    args = ap.parse_args()

    print(f"Merging FSDP shards from {args.actor_dir} ...", flush=True)
    merged = _merge_fsdp_state_dict(args.actor_dir)
    print(f"Merged state dict: {len(merged)} tensors", flush=True)

    template = torch.load(args.template_ckpt, map_location="cpu", weights_only=True)
    target_keys = list(template.keys())
    print(f"Target trainable params (from template): {len(target_keys)}", flush=True)

    out, missing, mismatched = {}, [], []
    for tk in target_keys:
        sk = _rl_source_key(tk)
        if sk is None or sk not in merged:
            missing.append((tk, sk))
            continue
        tensor = merged[sk]
        target_shape = tuple(template[tk].shape)
        if tuple(tensor.shape) != target_shape:
            # Scalars are stored 1-D in the GRPO checkpoint (unsqueezed for
            # FSDP/vLLM); reshape back when the element count matches.
            if tensor.numel() == template[tk].numel():
                tensor = tensor.reshape(target_shape)
            else:
                mismatched.append((tk, tuple(tensor.shape), target_shape))
                continue
        out[tk] = tensor.to(torch.float32).contiguous().clone()

    print(f"Converted {len(out)}/{len(target_keys)} params", flush=True)
    if missing:
        print(f"  MISSING {len(missing)} (first 5): {missing[:5]}")
    if mismatched:
        print(f"  SHAPE MISMATCH {len(mismatched)} (first 5): {mismatched[:5]}")
    if missing or mismatched:
        raise SystemExit("Conversion incomplete — not writing checkpoint.")

    dest_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "pytorch_model.pt")
    torch.save(out, dest)
    print(f"Wrote {dest} ({len(out)} params)", flush=True)


if __name__ == "__main__":
    main()
