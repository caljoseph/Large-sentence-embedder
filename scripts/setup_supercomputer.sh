#!/bin/bash

# Setup script for running the sentence embedding project on a supercomputer
# This script helps with installing dependencies and setting up the environment

# Create a virtual environment
echo "Creating a Python virtual environment..."
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust version as needed for your CUDA version)
echo "Installing PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Install DeepSpeed (optional, can be customized based on system)
echo "Installing DeepSpeed..."
# Special arguments might be needed for some systems
DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install deepspeed==0.10.0

# Log in to HuggingFace (required for Llama models)
echo "Please log in to HuggingFace to access gated models like Llama..."
huggingface-cli login

# Set up WANDB for logging (optional)
echo "Please set up your WANDB API key (skip if not using WANDB)..."
echo "Run: wandb login"

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.sh

# Create directories for outputs
echo "Creating output directories..."
mkdir -p checkpoints
mkdir -p evaluation_results

echo "Setup complete! You can now submit jobs using the Slurm scripts, for example:"
echo "  sbatch scripts/slurm_train.sh"
echo "  sbatch scripts/slurm_train_lora.sh"
echo "  sbatch scripts/slurm_train_multinode.sh"
echo "  sbatch scripts/slurm_evaluate.sh"