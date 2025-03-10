#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --response_length)
            RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --tp_size)
            TP_SIZE="$2"
            shift 2
            ;;
        --rollouts)
            ROLLOUTS="$2"
            shift 2
            ;;
        --val_rollouts)
            VAL_ROLLOUTS="$2"
            shift 2
            ;;
        --gpu_mem_util)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --batch_size)
            OVERRIDE_BATCH_SIZE="$2"
            shift 2
            ;;
        --correct_ids_path)
            CORRECT_IDS_PATH="$2"
            shift 2
            ;;
        --project_name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --s3_bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --s3_path)
            S3_PATH="$2"
            shift 2
            ;;
        --aws_region)
            AWS_REGION="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="Qwen/Qwen2.5-14B"
fi

# Set default number of nodes
if [ -z "$NNODES" ]; then
    NNODES=2
fi

# Set default response length for context (20K by default)
if [ -z "$RESPONSE_LENGTH" ]; then
    RESPONSE_LENGTH=20480
fi

# Set default tensor parallel size
if [ -z "$TP_SIZE" ]; then
    TP_SIZE=4  # Good default for 14B models
fi

# Set default number of rollouts
# For TP_SIZE=4 with 2 nodes (16 GPUs), we can do 4 model instances with 4 GPUs each
# So each model instance can generate these many rollouts in parallel
if [ -z "$ROLLOUTS" ]; then
    # Default rollouts based on TP size to optimize throughput
    if [ $TP_SIZE -eq 8 ]; then
        ROLLOUTS=4  # With 8-way TP, we have fewer model instances
    elif [ $TP_SIZE -eq 4 ]; then
        ROLLOUTS=8  # With 4-way TP, we have a good balance
    else
        ROLLOUTS=16  # With lower TP, we can do more rollouts
    fi
fi

# Set default validation rollouts (usually same as training rollouts)
if [ -z "$VAL_ROLLOUTS" ]; then
    VAL_ROLLOUTS=$ROLLOUTS
fi

# Set default GPU memory utilization (0.85 is a safe default for most setups)
if [ -z "$GPU_MEM_UTIL" ]; then
    GPU_MEM_UTIL=0.85
fi

# Set default project name
if [ -z "$PROJECT_NAME" ]; then
    PROJECT_NAME="deepscaler"
fi

# Set default experiment name
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="deepscaler-14b-arc-agi-${RESPONSE_LENGTH}-tp${TP_SIZE}"
fi

# Configure S3 URI if provided
if [ ! -z "$S3_BUCKET" ] && [ ! -z "$S3_PATH" ]; then
    if [ ! -z "$AWS_REGION" ]; then
        export AWS_DEFAULT_REGION="$AWS_REGION"
    fi
    S3_URI="s3://${S3_BUCKET}/${S3_PATH}"
    echo "Checkpoints will be saved to: $S3_URI"
    CHECKPOINT_DIR=$S3_URI
else
    CHECKPOINT_DIR="null"  # Default is not to save checkpoints externally
fi

# Generate the ARG-AGI dataset files if they don't exist
if [ ! -f "$HOME/deepscaler/data/arc_agi_train.parquet" ] || [ ! -f "$HOME/deepscaler/data/arc_agi_test.parquet" ]; then
    echo "Processing ARG-AGI dataset..."
    CORRECT_IDS_ARG=""
    if [ ! -z "$CORRECT_IDS_PATH" ]; then
        CORRECT_IDS_ARG="--correct_ids_path $CORRECT_IDS_PATH"
    fi
    python3 "$HOME/deepscaler/scripts/data/arc_dataset.py" $CORRECT_IDS_ARG
fi

# Calculate optimal PPO batch sizes based on context length
# These values might need adjustment based on your specific model and hardware
if [ ! -z "$OVERRIDE_BATCH_SIZE" ]; then
    echo "Using overridden batch size: $OVERRIDE_BATCH_SIZE"
    TRAIN_BATCH_SIZE=$OVERRIDE_BATCH_SIZE
    VAL_BATCH_SIZE=$((OVERRIDE_BATCH_SIZE * 2))
    PPO_MINI_BATCH_SIZE=$((OVERRIDE_BATCH_SIZE / 2))
elif [ $RESPONSE_LENGTH -gt 16384 ]; then
    TRAIN_BATCH_SIZE=64  # Smaller batch for very long contexts
    VAL_BATCH_SIZE=128
    PPO_MINI_BATCH_SIZE=32
elif [ $RESPONSE_LENGTH -gt 8192 ]; then
    TRAIN_BATCH_SIZE=96  # Medium batch for long contexts
    VAL_BATCH_SIZE=192
    PPO_MINI_BATCH_SIZE=48
else
    TRAIN_BATCH_SIZE=128  # Standard batch for normal contexts
    VAL_BATCH_SIZE=256
    PPO_MINI_BATCH_SIZE=64
fi

# Ensure mini batch size is at least 1
if [ $PPO_MINI_BATCH_SIZE -lt 1 ]; then
    PPO_MINI_BATCH_SIZE=1
fi

echo "Configuration summary:"
echo "Model: $MODEL_PATH"
echo "Nodes: $NNODES"
echo "Response length: $RESPONSE_LENGTH"
echo "Tensor parallel size: $TP_SIZE"
echo "Rollouts: $ROLLOUTS"
echo "Validation rollouts: $VAL_ROLLOUTS"
echo "GPU memory utilization: $GPU_MEM_UTIL"
echo "Training batch size: $TRAIN_BATCH_SIZE"
echo "Validation batch size: $VAL_BATCH_SIZE"
echo "PPO mini batch size: $PPO_MINI_BATCH_SIZE"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Project name: $PROJECT_NAME"

# Train over specified nodes, 8 H100 GPUs per node
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/deepscaler/data/arc_agi_train.parquet \
    data.val_files=$HOME/deepscaler/data/arc_agi_test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=$RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=26214 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$TP_SIZE \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUTS \
    actor_rollout_ref.rollout.n_val=$VAL_ROLLOUTS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=$CHECKPOINT_DIR \
    trainer.total_epochs=30 \
    +actor_rollout_ref.model.torch_dtype=bf16 "${@:1}"