#!/usr/bin/env python3
"""
Create the unbaked HF model directory for TSLLMForCausalLM.

Downloads Qwen3-4B base weights + tokenizer from HuggingFace, then copies
the custom model code (modeling_tsllm.py → hf_tsllm.py) and config.json
so that AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
works.

Usage:
    python3 -m teleprism.rl.training.setup_model_dir --output-dir /path/to/tsllm-model-dir
"""

import argparse
import json
import os
import shutil

def main():
    parser = argparse.ArgumentParser(description="Create unbaked HF model directory")
    parser.add_argument("--output-dir", required=True, help="Output HF model directory")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B", help="HF base model ID")
    args = parser.parse_args()

    output_dir = args.output_dir

    if os.path.isfile(os.path.join(output_dir, "config.json")):
        print(f"Model directory already exists at {output_dir}, skipping setup.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Download base model (safetensors + tokenizer)
    print(f"Downloading {args.base_model} to {output_dir}...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        args.base_model,
        local_dir=output_dir,
        ignore_patterns=["*.gguf", "*.bin"],  # skip non-safetensors formats
    )
    print(f"  Download complete.")

    # Copy custom model code as hf_tsllm.py (trust_remote_code entry point)
    src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ts_llm", "tsllm_causal")
    src_dir = os.path.normpath(src_dir)

    modeling_src = os.path.join(src_dir, "modeling_tsllm.py")
    hf_tsllm_dst = os.path.join(output_dir, "hf_tsllm.py")
    shutil.copy2(modeling_src, hf_tsllm_dst)
    print(f"  Copied modeling_tsllm.py → hf_tsllm.py")

    # Write custom config.json (merging base config with our overrides)
    base_config_path = os.path.join(output_dir, "config.json")
    with open(base_config_path) as f:
        config = json.load(f)

    # Only overlay the fields we need (auto_map, architectures, lora_alpha).
    # Do NOT override model architecture fields (vocab_size, hidden_size, etc.)
    # from the template — those belong to the base model.
    tsllm_config_path = os.path.join(src_dir, "config.json")
    with open(tsllm_config_path) as f:
        tsllm_overrides = json.load(f)

    OVERLAY_KEYS = {"architectures", "auto_map", "lora_alpha"}
    for key in OVERLAY_KEYS:
        if key in tsllm_overrides:
            config[key] = tsllm_overrides[key]

    with open(base_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Updated config.json with TSLLM auto_map and settings")

    print(f"\nUnbaked model directory ready: {output_dir}")


if __name__ == "__main__":
    main()
