#!/bin/bash

#SBATCH --time=47:59:59   # walltime - extend for multi-node training
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks) - one per node
#SBATCH --nodes=4   # number of nodes - adjust as needed
#SBATCH --ntasks-per-node=1  # one task per node
#SBATCH --gpus-per-node=8   # 8 GPUs per node for H100 machines
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH -J "sent-embed-H100"   # job name
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --qos=cs
##SBATCH --partition=cs # for access to a100s
#SBATCH --partition=cs2 # for access to h100s - specifically targeting H100s

# Set the max number of threads to use for programs using OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Load necessary modules (modify as needed for your environment)
# module load python/3.10
# module load cuda/11.8

# Print node information for debugging
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_NODEID: $SLURM_NODEID"
hostname
nvidia-smi

# Activate virtual environment if using one
# source /path/to/your/venv/bin/activate

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Number of GPUs per node - hardcoded to 8 for H100 nodes
export NUM_GPUS_PER_NODE=8

# Set up DeepSpeed parameters - optimize for H100s
DEEPSPEED_CONFIG="configs/deepspeed_config_h100.json" # Make sure this file exists with H100 optimizations
ENCODER_MODEL="meta-llama/Llama-2-13b-hf"  # Larger model for H100s
DECODER_MODEL="meta-llama/Llama-2-13b-hf"  # Larger model for H100s
CORPUS="c4"
SUBSET="en"
NUM_SAMPLES=10000000  # More samples for multi-node training
NUM_EPOCHS=3
MAX_LENGTH=256  # Longer sequences for better embedding quality
OUTPUT_DIR="checkpoints/h100_multinode_run"
BATCH_SIZE=32  # Larger batch size possible with H100s
WANDB_PROJECT="sentence-embed-h100-multinode"

# Create output directory
mkdir -p $OUTPUT_DIR

# Get the node rank from Slurm
NODE_RANK=$SLURM_NODEID

echo "Running multi-node training with $SLURM_NNODES nodes"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Launch with DeepSpeed
deepspeed --num_nodes=$SLURM_NNODES \
          --num_gpus=$NUM_GPUS_PER_NODE \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          --node_rank=$NODE_RANK \
          --deepspeed_config=$DEEPSPEED_CONFIG \
          scripts/train.py \
          --encoder_model_name $ENCODER_MODEL \
          --decoder_model_name $DECODER_MODEL \
          --corpus $CORPUS \
          --subset $SUBSET \
          --num_samples $NUM_SAMPLES \
          --batch_size $BATCH_SIZE \
          --num_epochs $NUM_EPOCHS \
          --max_length $MAX_LENGTH \
          --output_dir $OUTPUT_DIR \
          --use_deepspeed \
          --wandb \
          --wandb_project $WANDB_PROJECT

echo "Multi-node training job completed"