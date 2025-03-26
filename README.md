# Sentence Embedding Autoencoder

This repository contains the implementation of a sentence embedding autoencoder model designed for generating high-quality, fixed-size sentence representations. The model uses a transformer encoder (e.g., Llama-2) to create an initial embedding and a transformer decoder to reconstruct the original sentence from this embedding combined with learned "blank" tokens. This encourages the single embedding vector to capture the core semantic meaning necessary for reconstruction.

The project supports:
*   Standard fine-tuning.
*   Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation).
*   Distributed training using PyTorch DDP or DeepSpeed (including ZeRO optimizations).
*   Comprehensive evaluation on STS, classification, and clustering benchmarks.
*   Comparison against baseline models from `sentence-transformers`.

## Project Structure

```
sentence-embedding-autoencoder/
├── model/
│   └── sentence_embedding_model.py   # Core model implementation (Standard + LoRA)
├── scripts/
│   ├── train.py                      # Main training script
│   ├── evaluate.py                   # Main evaluation script
│   └── run_distributed.sh            # Launcher for distributed training
├── configs/
│   └── deepspeed_config.json         # Example DeepSpeed configuration (ZeRO-3)
├── data_and_training.py              # Data loading, Dataset class, training loops
├── evaluation_suite.py               # Evaluator class, comparison logic, plotting
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd sentence-embedding-autoencoder
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Install PyTorch:**
    Follow instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) for your specific CUDA version (required for GPU training).

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install DeepSpeed (Optional, for advanced distributed training):**
    Follow the official [DeepSpeed installation guide](https://www.deepspeed.ai/getting-started/#installation) for your system and CUDA version. This often involves C++ extensions.
    ```bash
    # Example (may vary):
    pip install deepspeed
    ```

6.  **Login to Hugging Face Hub (if using gated models like Llama-2):**
    ```bash
    huggingface-cli login
    ```

## Usage

### Training

The `scripts/train.py` script handles training. Key arguments include model names, data configuration, training parameters, and distributed settings.

**Command Structure:**

```bash
# For DDP (use run_distributed.sh):
bash scripts/run_distributed.sh --num_gpus <N> [scripts/train.py arguments...]

# For DeepSpeed (use run_distributed.sh):
bash scripts/run_distributed.sh --num_gpus <N> --deepspeed [--deepspeed_config <path>] [scripts/train.py arguments...]

# For single GPU/CPU (direct execution):
python scripts/train.py [arguments...]
```

**Examples:**

*   **Single GPU Training (Llama-2 7B):**
    ```bash
    python scripts/train.py \
        --encoder_model_name meta-llama/Llama-2-7b-hf \
        --decoder_model_name meta-llama/Llama-2-7b-hf \
        --corpus c4 --subset en --num_samples 1000000 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_epochs 3 \
        --max_length 128 \
        --pooling_strategy last_token \
        --output_dir checkpoints/llama7b_c4 \
        --fp16 \
        --wandb --wandb_project "sentence-embed-llama"
    ```

*   **Multi-GPU Training (8 GPUs) with DDP:**
    ```bash
    bash scripts/run_distributed.sh --num_gpus 8 \
        --encoder_model_name meta-llama/Llama-2-7b-hf \
        --decoder_model_name meta-llama/Llama-2-7b-hf \
        --corpus c4 --subset en --num_samples 5000000 \
        --batch_size 4 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-5 \
        --num_epochs 3 \
        --max_length 128 \
        --output_dir checkpoints/llama7b_c4_ddp \
        --fp16 \
        --wandb --wandb_project "sentence-embed-llama"
    ```

*   **Multi-GPU Training (8 GPUs) with DeepSpeed ZeRO-3:** (Recommended for large models)
    ```bash
    bash scripts/run_distributed.sh --num_gpus 8 --deepspeed --deepspeed_config configs/deepspeed_config.json \
        --encoder_model_name meta-llama/Llama-2-7b-hf \
        --decoder_model_name meta-llama/Llama-2-7b-hf \
        --corpus c4 --subset en --num_samples 10000000 \
        # batch_size, grad_accum, fp16 are often controlled by deepspeed config
        --learning_rate 2e-5 # Can be overridden here or set in config
        --num_epochs 3 \
        --max_length 128 \
        --output_dir checkpoints/llama7b_c4_ds_z3 \
        --wandb --wandb_project "sentence-embed-llama"
    ```
    *Note: Adjust batch size, learning rate, etc., in `configs/deepspeed_config.json` or via command line.*

*   **Parameter-Efficient Training with LoRA (Single GPU):**
    ```bash
    python scripts/train.py \
        --encoder_model_name meta-llama/Llama-2-7b-hf \
        --decoder_model_name meta-llama/Llama-2-7b-hf \
        --use_lora --lora_r 16 --lora_alpha 32 \
        --corpus c4 --subset en --num_samples 1000000 \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4 \
        --num_epochs 3 \
        --max_length 128 \
        --output_dir checkpoints/llama7b_lora_c4 \
        --fp16 \
        --wandb --wandb_project "sentence-embed-llama-lora"
    ```

### Evaluation

The `scripts/evaluate.py` script loads a trained model (yours or a baseline) and runs it through the evaluation suite (`evaluation_suite.py`).

**Example:**

```bash
python scripts/evaluate.py \
    --model_path checkpoints/llama7b_c4_ds_z3/best_model.pt \
    --model_name "MyLlama7B_DS_Z3" \
    # Specify base model details if loading a raw state dict
    --encoder_model_name meta-llama/Llama-2-7b-hf \
    --decoder_model_name meta-llama/Llama-2-7b-hf \
    # Add --use_lora if evaluating a LoRA checkpoint
    --tasks all \
    --compare_with sentence-transformers/all-mpnet-base-v2 sentence-transformers/all-MiniLM-L6-v2 \
    --batch_size 64 \
    --output_dir evaluation_results/my_model_vs_baselines \
    --wandb --wandb_project "sentence-embed-eval"
```

Evaluation results (JSON files, plots) will be saved to the specified `--output_dir`.

### Training on Supercomputers (e.g., 8x H100)

For large-scale training on multi-GPU systems like those with H100s, **DeepSpeed with ZeRO Stage 3** is highly recommended.

1.  **Configure DeepSpeed:** Edit `configs/deepspeed_config.json`. Pay attention to:
    *   `train_micro_batch_size_per_gpu`: Set based on GPU memory (e.g., 8, 16, 32 for H100s).
    *   `gradient_accumulation_steps`: Adjust to reach your desired effective batch size (e.g., `micro_batch * num_gpus * grad_accum = 1024` or higher).
    *   `optimizer` and `scheduler` parameters (learning rate, warmup steps, total steps). Ensure `total_num_steps` in the scheduler matches your estimated training duration.
    *   `fp16.enabled`: Should be `true` for H100 performance.
    *   `zero_optimization.stage`: Should be `3`. Offloading options can save memory but may slow down training slightly.

2.  **Launch using `run_distributed.sh`:**
    ```bash
    bash scripts/run_distributed.sh \
        --num_gpus 8 \
        --deepspeed \
        --deepspeed_config configs/deepspeed_config.json \
        \
        # --- Pass relevant train.py arguments ---
        --encoder_model_name meta-llama/Llama-2-13b-hf # Example: Larger model
        --decoder_model_name meta-llama/Llama-2-13b-hf \
        --corpus openwebtext --num_samples 50000000 # Larger dataset
        --num_epochs 2 \
        --max_length 256 # Longer sequences if needed
        --output_dir checkpoints/llama13b_owt_ds_z3 \
        --logging_steps 50 \
        --wandb --wandb_project "sentence-embed-large" \
        # Add any other relevant args for train.py
    ```

3.  **Monitor:** Use tools like `nvidia-smi` and WandB (if enabled) to monitor GPU utilization, memory usage, and training progress. Adjust DeepSpeed config (batch size, grad accum) if you encounter OOM errors or low utilization.

## License

This project is licensed under the MIT License. See the LICENSE file for details.