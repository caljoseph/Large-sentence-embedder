#!/bin/bash

#SBATCH --time=23:59:59   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8   # number of GPUs
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH -J "sentence-embed"   # job name
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --qos=cs
#SBATCH --partition=cs # for access to a100s
##SBATCH --partition=cs2 # for access to h100s (uncomment to use)

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
hostname
nvidia-smi

# Activate virtual environment if using one
# source /path/to/your/venv/bin/activate

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Number of GPUs per node (assuming all nodes have the same number)
export NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=count --format=csv,noheader)

# Set up DeepSpeed parameters
DEEPSPEED_CONFIG="configs/deepspeed_config.json"
ENCODER_MODEL="meta-llama/Llama-2-7b-hf"  # Change as needed
DECODER_MODEL="meta-llama/Llama-2-7b-hf"  # Change as needed
CORPUS="c4"                          # Change as needed
SUBSET="en"                          # Change as needed
NUM_SAMPLES=1000000                  # Change as needed
NUM_EPOCHS=3                         # Change as needed
MAX_LENGTH=128                       # Change as needed
OUTPUT_DIR="checkpoints/slurm_run"   # Change as needed
BATCH_SIZE=8                         # This will be used if not specified in deepspeed config
WANDB_PROJECT="sentence-embed-supercomputer" # Change as needed

# Create output directory
mkdir -p $OUTPUT_DIR

# If using multiple nodes, adjust the distributed training parameters
if [ "$SLURM_NNODES" -gt 1 ]; then
    # Get the node rank from Slurm (assumes one task per node)
    NODE_RANK=$SLURM_NODEID
    
    echo "Running multi-node training with $SLURM_NNODES nodes"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "NODE_RANK: $NODE_RANK"
    
    # Launch with DeepSpeed
    deepspeed --num_nodes=$SLURM_NNODES \
              --num_gpus=$NUM_GPUS_PER_NODE \
              --master_addr=$MASTER_ADDR \
              --master_port=$MASTER_PORT \
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
else
    # Single node, multiple GPUs
    echo "Running single-node training with $NUM_GPUS_PER_NODE GPUs"
    
    # Launch with DeepSpeed
    deepspeed --num_gpus=$NUM_GPUS_PER_NODE \
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
fi

echo "Training job completed"