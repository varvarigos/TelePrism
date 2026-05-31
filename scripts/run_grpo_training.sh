#!/bin/bash
#
# TelecomTS QA GRPO Training - Qwen3-4B with Time-Series Fusion
#
# All configuration lives in configs/grpo_tsllm.yaml.
# This script is a thin launcher that reads the YAML and invokes VERL.
#
# Usage:
#   bash scripts/run_grpo_training.sh [additional hydra overrides]
#
# Override config file:
#   GRPO_CONFIG=configs/grpo_tsllm_toto.yaml bash scripts/run_grpo_training.sh
#
# Requires an activated .venv with `pip install -e .` + requirements.txt and a
# CUDA toolchain. `verl` is installed as a normal dependency (no vendored copy).

set -e
set -x
source .venv/bin/activate

# === CONFIG ===
GRPO_CONFIG=${GRPO_CONFIG:-"configs/grpo_tsllm.yaml"}
export GRPO_TSLLM_CONFIG="$GRPO_CONFIG"

# Helper: read a YAML key via Python (dot-separated path, e.g. "training.optimizer.learning_rate")
yq() { python3 -c "
import yaml, functools
c = yaml.safe_load(open('$GRPO_CONFIG'))
val = functools.reduce(lambda d, k: d[k], '$1'.split('.'), c)
print(val)
"; }

# === READ CONFIG ===
# Paths (BASE_DIR is the repo root = current working directory)
BASE_DIR="$(pwd)"
MNT_DIR=$(yq paths.data_dir)
MODEL_PATH=$(yq paths.model_path)
UNBAKED_DIR=$(yq paths.unbaked_model_dir)

# Starting model
STARTING_MODEL=$(yq starting_model)
export TS_CHECKPOINT_DIR=$(yq checkpoints.$STARTING_MODEL.dir)
export TS_CHECKPOINT_TAG=$(yq checkpoints.$STARTING_MODEL.tag)

# GPU
NUM_GPUS=$(yq gpu.num_gpus)
TENSOR_PARALLEL_SIZE=$(yq gpu.tensor_parallel_size)
NUM_AGENT_WORKERS=$((NUM_GPUS / TENSOR_PARALLEL_SIZE))

# Training
EXP_NAME=$(yq training.experiment_name)
TOTAL_EPOCHS=$(yq training.total_epochs)
GRAD_CLIP=$(yq training.grad_clip)
LEARNING_RATE=$(yq training.optimizer.learning_rate)
LR_SCHEDULER_TYPE=$(yq training.optimizer.lr_scheduler_type)
LR_WARMUP_STEPS=$(yq training.optimizer.lr_warmup_steps)
TRAIN_BATCH_SIZE=$(yq training.batch.train_batch_size)
PPO_MINI_BATCH_SIZE=$(yq training.batch.ppo_mini_batch_size)
PPO_MICRO_BATCH_SIZE=$(yq training.batch.ppo_micro_batch_size)
MAX_PROMPT_LENGTH=$(yq training.sequence.max_prompt_length)
MAX_RESPONSE_LENGTH=$(yq training.sequence.max_response_length)
ROLLOUT_N=$(yq training.grpo.rollout_n)
ROLLOUT_TEMP=$(yq training.grpo.temperature)
KL_LOSS_COEF=$(yq training.grpo.kl_loss_coef)
ENTROPY_COEFF=$(yq training.grpo.entropy_coeff)
GPU_MEM_UTIL=$(yq training.rollout.gpu_memory_utilization)
MAX_MODEL_LEN=$(yq training.rollout.max_model_len)
MAX_NUM_SEQS=$(yq training.rollout.max_num_seqs)

# LoRA
USE_LORA=$(yq lora.enabled)
LORA_RANK=$(yq lora.rank)
LORA_ALPHA=$(yq lora.alpha)
LORA_TARGET_MODULES=$(yq lora.target_modules)
LORA_EXCLUDE_MODULES=$(yq lora.exclude_modules)

# === ENVIRONMENT ===
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Use gcc/g++ 11 for CUDA extension builds when available (some CUDA versions
# reject newer host compilers); otherwise fall back to the system default.
if command -v g++-11 >/dev/null 2>&1; then
    export CUDAHOSTCXX=g++-11
    export CC=gcc-11
    export CXX=g++-11
fi
# export WANDB_API_KEY=<your-wandb-api-key>

# Derived paths
VERL_DIR="$BASE_DIR/src/rl/verl"           # vendored verl (used directly, not pip-installed)
OUTPUT_DIR="$MNT_DIR/checkpoints/$EXP_NAME"
VAL_DATA_DIR="$MNT_DIR/val/$EXP_NAME"
ROLLOUT_DATA_DIR="$MNT_DIR/rollout/$EXP_NAME"
REWARD_FUNCTION_PATH="$BASE_DIR/src/rl/training/rewards/reward_telecom.py"
TOKENIZER_DIR="$MODEL_PATH"

# Put both the repo root (for `teleprism`) and the vendored verl on the path.
export PYTHONPATH="$BASE_DIR:$VERL_DIR:$PYTHONPATH"
export VERL_USE_EXTERNAL_MODULES="teleprism.rl.training.agent_loop,teleprism.rl.training.fsdp_patches"

# === AUTO-SETUP ===
# Step 1: Create unbaked HF model dir if it doesn't exist
if [ ! -f "$UNBAKED_DIR/config.json" ]; then
    echo "═══════════════════════════════════════════════"
    echo " Unbaked model dir not found at $UNBAKED_DIR"
    echo " Downloading Qwen3-4B and setting up model dir..."
    echo "═══════════════════════════════════════════════"
    python3 -m teleprism.rl.training.setup_model_dir \
        --output-dir "$UNBAKED_DIR" || {
        echo "ERROR: Model dir setup failed."
        exit 1
    }
    echo ""
fi

# Step 2: Bake (merge TS encoder + LoRA + base LLM) into HF model dir
# Each starting_model gets its own baked dir so you can switch without re-baking.
BAKED_DIR="${MODEL_PATH}-${STARTING_MODEL}"
if [ "$STARTING_MODEL" = "cold_start" ]; then
    # Keep backward-compatible dir name for cold_start
    BAKED_DIR="$MODEL_PATH"
fi

if [ ! -d "$BAKED_DIR" ]; then
    echo "═══════════════════════════════════════════════"
    echo " Baking model (starting_model=$STARTING_MODEL)"
    echo " Checkpoint: $TS_CHECKPOINT_DIR/$TS_CHECKPOINT_TAG"
    echo " Output:     $BAKED_DIR"
    echo "═══════════════════════════════════════════════"
    python3 -m teleprism.rl.training.bake_checkpoint \
        --model-dir "$UNBAKED_DIR" \
        --output-dir "$BAKED_DIR" || {
        echo "ERROR: Baking failed."
        exit 1
    }
    echo " Baking complete → $BAKED_DIR"
    echo ""
fi
MODEL_PATH="$BAKED_DIR"
TOKENIZER_DIR="$BAKED_DIR"

# Step 3: Always sync modeling_tsllm.py → hf_tsllm.py in model dirs
#          (HF caches the trust_remote_code module, so we must keep it fresh)
MODELING_SRC="$BASE_DIR/src/models/ts_llm/tsllm_causal/modeling_tsllm.py"
for dir in "$MODEL_PATH" "$UNBAKED_DIR"; do
    if [ -d "$dir" ]; then
        cp "$MODELING_SRC" "$dir/hf_tsllm.py"
    fi
done
rm -rf ~/.cache/huggingface/modules/transformers_modules/tsllm_hyphen_model_hyphen_dir_hyphen_baked/ 2>/dev/null
echo " Synced hf_tsllm.py + cleared HF cache"

# === DATA ===
TRAIN_FILE="$MNT_DIR/data/telecom_train_grpo.parquet"
TEST_FILE="$MNT_DIR/data/telecom_test_grpo.parquet"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    echo "═══════════════════════════════════════════════"
    echo " Data files not found. Running preprocessing..."
    echo "═══════════════════════════════════════════════"
    python3 -m teleprism.rl.training.preprocessing.preprocess_telecom_data --output_dir "$MNT_DIR/data" --split train
    python3 -m teleprism.rl.training.preprocessing.preprocess_telecom_data --output_dir "$MNT_DIR/data" --split test
    echo " Data preprocessing complete"
    echo ""
fi

mkdir -p "$OUTPUT_DIR" "$VAL_DATA_DIR" "$ROLLOUT_DATA_DIR"

# === SUMMARY ===
echo "=============================================="
echo " TelecomTS QA GRPO Training"
echo "=============================================="
echo "  Config:       $GRPO_CONFIG"
echo "  Experiment:   $EXP_NAME"
echo "  Starting:     $STARTING_MODEL"
echo "  Checkpoint:   $TS_CHECKPOINT_DIR/$TS_CHECKPOINT_TAG"
echo "  Model:        $MODEL_PATH"
echo "  Encoder:      $(yq encoder)"
echo "  GPUs:         $NUM_GPUS (TP=$TENSOR_PARALLEL_SIZE)"
echo "  Batch:        $TRAIN_BATCH_SIZE (mini=$PPO_MINI_BATCH_SIZE, micro=$PPO_MICRO_BATCH_SIZE)"
echo "  Rollouts:     $ROLLOUT_N"
echo "  Epochs:       $TOTAL_EPOCHS"
echo "  LR:           $LEARNING_RATE ($LR_SCHEDULER_TYPE, warmup_steps=$LR_WARMUP_STEPS)"
echo "  KL Coef:      $KL_LOSS_COEF"
echo "  LoRA:         r=$LORA_RANK, alpha=$LORA_ALPHA"
echo "  Output:       $OUTPUT_DIR"
echo "=============================================="
echo ""

train_files="['$TRAIN_FILE']"
test_files="['$TEST_FILE']"

cd "$BASE_DIR"

# Build LoRA args conditionally
LORA_ARGS=()
if [ "$USE_LORA" = "True" ] || [ "$USE_LORA" = "true" ]; then
    LORA_ARGS=(
        "actor_rollout_ref.model.lora_rank=$LORA_RANK"
        "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
        "actor_rollout_ref.model.target_modules='$LORA_TARGET_MODULES'"
        "actor_rollout_ref.model.exclude_modules=$LORA_EXCLUDE_MODULES"
        "+actor_rollout_ref.model.lora.merge=True"
    )
    echo "  LoRA: ENABLED (r=$LORA_RANK, alpha=$LORA_ALPHA)"
else
    echo "  LoRA: DISABLED (full-parameter training)"
fi

python3 -m teleprism.rl.training.main \
    algorithm.adv_estimator=grpo \
    +algorithm.filter_groups.enable=True \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation='left' \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.tokenizer_path="$TOKENIZER_DIR" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.external_lib=teleprism.rl.training.rollout \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_scheduler_type=$LR_SCHEDULER_TYPE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    "${LORA_ARGS[@]}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=ts_vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=teleprism.rl.training.worker_extension.TSColocateWorkerExtension \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.timeseries=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=ts_single_turn_agent \
    actor_rollout_ref.rollout.agent.num_workers=$NUM_AGENT_WORKERS \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='telecom_qa' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.validation_data_dir="$VAL_DATA_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.resume_mode=disable \
    "$@"

echo ""
echo "=============================================="
echo " Training complete!"
echo " Experiment: $EXP_NAME"
echo " Checkpoints: $OUTPUT_DIR"
echo "=============================================="
