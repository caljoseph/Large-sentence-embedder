#!/bin/bash

# Distributed training launcher for sentence embedding autoencoder
# Usage:
#   - PyTorch DDP: bash run_distributed.sh --num_gpus 8 [train.py args...]
#   - DeepSpeed:   bash run_distributed.sh --num_gpus 8 --deepspeed --deepspeed_config configs/my_config.json [train.py args...]

# --- Default Configuration ---
NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=count --format=csv,noheader) # Autodetect GPUs
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500 # Default port, change if needed
USE_DEEPSPEED=0
DEEPSPEED_CONFIG=""
TRAIN_SCRIPT="scripts/train.py" # Relative path from project root

# --- Parse Launcher Arguments ---
# Separate launcher args from script args
LAUNCHER_ARGS=()
SCRIPT_ARGS=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --num_gpus)
        NUM_GPUS_PER_NODE="$2"
        shift; shift
        ;;
        --num_nodes)
        NUM_NODES="$2"
        shift; shift
        ;;
        --node_rank)
        NODE_RANK="$2"
        shift; shift
        ;;
        --master_addr)
        MASTER_ADDR="$2"
        shift; shift
        ;;
        --master_port)
        MASTER_PORT="$2"
        shift; shift
        ;;
        --deepspeed)
        USE_DEEPSPEED=1
        shift
        ;;
        --deepspeed_config)
        DEEPSPEED_CONFIG="$2"
        shift; shift
        ;;
        --train_script)
        TRAIN_SCRIPT="$2"
        shift; shift
        ;;
        *)  # Assume remaining arguments are for the training script
        SCRIPT_ARGS+=("$1")
        shift
        ;;
    esac
done

# --- Validate Configuration ---
if ! command -v python &> /dev/null; then
    echo "Error: python command not found. Make sure Python is installed and in your PATH."
    exit 1
fi

if [ $USE_DEEPSPEED -eq 1 ] && ! command -v deepspeed &> /dev/null; then
    echo "Error: --deepspeed specified but deepspeed command not found. Install DeepSpeed."
    exit 1
fi


# --- Print Configuration ---
echo "-------------------- Distributed Training Launcher --------------------"
echo "Timestamp: $(date)"
echo "Project Root: $(pwd)" # Assuming script is run from project root
echo "Node Rank: $NODE_RANK / $NUM_NODES"
echo "GPUs per Node: $NUM_GPUS_PER_NODE"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Using DeepSpeed: $USE_DEEPSPEED"
[ $USE_DEEPSPEED -eq 1 ] && [ -n "$DEEPSPEED_CONFIG" ] && echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "Training Script: $TRAIN_SCRIPT"
echo "Script Arguments: ${SCRIPT_ARGS[@]}"
echo "-----------------------------------------------------------------------"
echo ""

# --- Set Environment Variables for PyTorch Distributed ---
# These are used by both torch.distributed.launch and DeepSpeed launcher
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# NCCL debugging (optional)
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0 # Adjust to your network interface if needed
# export NCCL_IB_DISABLE=1 # Disable InfiniBand if causing issues


# --- Launch Training ---
echo "Launching training..."

if [ $USE_DEEPSPEED -eq 1 ]; then
    # Launch with DeepSpeed launcher
    # It handles setting RANK, WORLD_SIZE, LOCAL_RANK etc. internally
    LAUNCH_CMD="deepspeed --num_nodes=$NUM_NODES --num_gpus=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"

    # Add hostfile argument if running multi-node without resource manager (e.g., Slurm/MPI)
    # Example: --hostfile path/to/hostfile (where hostfile lists IP addresses/hostnames)

    # Add DeepSpeed config if specified
    if [ -n "$DEEPSPEED_CONFIG" ]; then
        LAUNCH_CMD+=" --deepspeed_config $DEEPSPEED_CONFIG"
    fi

    # Add the training script and its arguments
    # Ensure the script knows it's using deepspeed (passed via script args)
    LAUNCH_CMD+=" $TRAIN_SCRIPT --use_deepspeed ${SCRIPT_ARGS[@]}"

    echo "Executing DeepSpeed command:"
    echo "$LAUNCH_CMD"
    eval "$LAUNCH_CMD"
    EXIT_CODE=$?

else
    # Launch with torch.distributed.launch (deprecated) or torchrun (preferred)
    # torchrun is generally recommended for PyTorch >= 1.10
    if command -v torchrun &> /dev/null; then
        LAUNCHER="torchrun"
    elif command -v python -m torch.distributed.launch &> /dev/null; then
         LAUNCHER="python -m torch.distributed.launch"
         echo "Warning: torch.distributed.launch is deprecated. Consider using torchrun."
    else
         echo "Error: Neither torchrun nor torch.distributed.launch found."
         exit 1
    fi


    LAUNCH_CMD="$LAUNCHER --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"

    # Add the training script and its arguments
    # --distributed flag is not needed for torchrun/launch, they set env vars
    # The script should detect distributed mode via env vars (handled by setup_distributed_training)
    LAUNCH_CMD+=" $TRAIN_SCRIPT ${SCRIPT_ARGS[@]}"

    echo "Executing PyTorch Distributed command:"
    echo "$LAUNCH_CMD"
    eval "$LAUNCH_CMD"
    EXIT_CODE=$?
fi

echo "-----------------------------------------------------------------------"
echo "Training script finished with exit code: $EXIT_CODE"
echo "-----------------------------------------------------------------------"
exit $EXIT_CODE