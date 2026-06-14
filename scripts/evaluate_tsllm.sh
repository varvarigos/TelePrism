#!/bin/bash
set -e

# Multi-task evaluation over the 7 TelecomTS tasks.

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Weights & Biases: run `wandb login` once, or set WANDB_API_KEY in your shell.
# Use gcc/g++ 11 for CUDA extension builds when available (some CUDA versions
# reject newer host compilers); otherwise fall back to the system default.
if command -v g++-11 >/dev/null 2>&1; then
    export CUDAHOSTCXX=g++-11
    export CC=gcc-11
    export CXX=g++-11
fi

deepspeed src/evaluate_tsllm.py \
    --checkpoint_dir ./checkpoints/cold_start \
    --tag epoch-40-6340 \
    --batch_size 128 \
    --deepspeed_config configs/ds_conf.json \
    --llm_model Qwen/Qwen3-4B \
    --lora_r 16 \
    --predictions_dir ./predictions \
    --save_predictions True \
    --sample_size 5
