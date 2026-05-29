#!/bin/bash
set -e
source .venv/bin/activate

# Multi-task evaluation over the 7 TelecomTS tasks.
#
# NOTE: `source .venv/bin/activate` is required so the DeepSpeed launcher uses
# the venv python in its worker subprocesses (see scripts/train_tsllm.sh).

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export WANDB_API_KEY=<your-wandb-api-key>
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDAHOSTCXX=g++-11
export CC=gcc-11
export CXX=g++-11

deepspeed --master_port=29501 --include localhost:2,3 src/evaluate_tsllm.py \
    --checkpoint_dir ./checkpoints/cold_start \
    --tag epoch-17-1851 \
    --batch_size 128 \
    --deepspeed_config configs/ds_conf.json \
    --llm_model Qwen/Qwen3-4B \
    --lora_r 16 \
    --predictions_dir ./predictions \
    --save_predictions True \
    --sample_size 5
