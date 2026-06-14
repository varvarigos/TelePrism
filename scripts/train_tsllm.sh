#!/bin/bash
set -e

# Cold-start SFT of TelePrism (TeleEncoder + Qwen3-4B) with DeepSpeed Zero-3 + LoRA.

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# CUDA is auto-detected from your environment; only set CUDA_HOME if your
# toolkit isn't found (e.g. `export CUDA_HOME=/usr/local/cuda`).
# Weights & Biases: run `wandb login` once, or set WANDB_API_KEY in your shell.
# Use gcc/g++ 11 for CUDA extension builds when available (some CUDA versions
# reject newer host compilers); otherwise fall back to the system default.
if command -v g++-11 >/dev/null 2>&1; then
    export CUDAHOSTCXX=g++-11
    export CC=gcc-11
    export CXX=g++-11
fi

deepspeed src/train_tsllm.py \
    --deepspeed_config configs/ds_conf.json \
    --llm_model Qwen/Qwen3-4B \
    --lora_r 16 \
    --epochs 41
