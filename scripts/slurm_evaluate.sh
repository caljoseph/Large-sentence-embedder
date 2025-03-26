#!/bin/bash

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1   # number of GPUs - evaluation usually only needs 1
#SBATCH --mem-per-cpu=32G   # memory per CPU core - evaluation might need more memory for datasets
#SBATCH -J "sent-embed-eval"   # job name
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
hostname
nvidia-smi

# Activate virtual environment if using one
# source /path/to/your/venv/bin/activate

# Model path (change as needed)
MODEL_PATH="checkpoints/slurm_run/best_model.pt"  # Path to the trained model
MODEL_NAME="Sentence-Embed-Supercomputer"  # Name for results
ENCODER_MODEL="meta-llama/Llama-2-7b-hf"  # Base encoder model for loading state dict
DECODER_MODEL="meta-llama/Llama-2-7b-hf"  # Base decoder model for loading state dict
COMPARE_WITH="sentence-transformers/all-mpnet-base-v2 sentence-transformers/all-MiniLM-L6-v2"  # Baseline models to compare with
OUTPUT_DIR="evaluation_results/supercomputer_eval"
USE_LORA=""  # Add --use_lora if model was trained with LoRA
BATCH_SIZE=64  # Batch size for evaluation
WANDB_PROJECT="sentence-embed-eval-supercomputer"  # WandB project for logging

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation
python scripts/evaluate.py \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --encoder_model_name $ENCODER_MODEL \
    --decoder_model_name $DECODER_MODEL \
    $USE_LORA \
    --tasks sts classification clustering \
    --compare_with $COMPARE_WITH \
    --batch_size $BATCH_SIZE \
    --max_samples_cluster 2000 \
    --max_train_samples_classif 10000 \
    --output_dir $OUTPUT_DIR \
    --wandb \
    --wandb_project $WANDB_PROJECT

echo "Evaluation job completed"