#!/bin/bash
set -e
source .venv/bin/activate

# Cold-start SFT of TelePrism (TeleEncoder + Qwen3-4B) with DeepSpeed Zero-3 + LoRA.
#
# NOTE: the `source .venv/bin/activate` above is required. Without it the
# DeepSpeed launcher spawns worker subprocesses with the system /usr/bin/python3
# (no torch/deepspeed), failing with confusing import errors.

export MASTER_PORT=29515
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Set your own Weights & Biases key (or run `wandb login`):
# export WANDB_API_KEY=<your-wandb-api-key>
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDAHOSTCXX=g++-11
export CC=gcc-11
export CXX=g++-11

deepspeed --include localhost:0,1 src/train_tsllm.py \
    --deepspeed_config configs/ds_conf.json \
    --llm_model Qwen/Qwen3-4B \
    --lora_r 16 \
    --epochs 41
