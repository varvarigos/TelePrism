#!/bin/bash
set -e
source .venv/bin/activate

# Single-prompt inference with TelePrism (TeleEncoder + Qwen3-4B).
#
# NOTE: `source .venv/bin/activate` is required so the DeepSpeed launcher uses
# the venv python in its worker subprocesses (see scripts/train_tsllm.sh).

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export WANDB_API_KEY=<your-wandb-api-key>
export CUDAHOSTCXX=g++-11
export CC=gcc-11
export CXX=g++-11

deepspeed --include localhost:0,3 src/inference_tsllm.py \
    --checkpoint_dir ./checkpoints/cold_start \
    --tag epoch-0-step-24999 \
    --deepspeed_config configs/ds_conf.json \
    --llm_model Qwen/Qwen3-4B-Instruct-2507 \
    --lora_r 16
