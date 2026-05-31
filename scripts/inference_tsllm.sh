#!/bin/bash
set -e
source .venv/bin/activate

# Single-prompt inference with TelePrism (TeleEncoder + Qwen3-4B).

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export WANDB_API_KEY=<your-wandb-api-key>
# Use gcc/g++ 11 for CUDA extension builds when available (some CUDA versions
# reject newer host compilers); otherwise fall back to the system default.
if command -v g++-11 >/dev/null 2>&1; then
    export CUDAHOSTCXX=g++-11
    export CC=gcc-11
    export CXX=g++-11
fi

deepspeed --include localhost:0,1 src/inference_tsllm.py \
    --checkpoint_dir ./checkpoints/cold_start \
    --tag epoch-0-step-24999 \
    --deepspeed_config configs/ds_conf.json \
    --llm_model Qwen/Qwen3-4B-Instruct-2507 \
    --lora_r 16
