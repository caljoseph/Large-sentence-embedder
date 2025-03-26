#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a sentence embedding autoencoder on a large text corpus.

This script trains the SentenceEmbeddingAutoencoder model using a specified text corpus
and configuration. It supports distributed training across multiple GPUs using DeepSpeed
or PyTorch Distributed Data Parallel (DDP).
"""

import os
import sys
import torch
import logging
import argparse
from datetime import datetime
import json
import random
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import project modules AFTER setting path
from model.sentence_embedding_model import SentenceEmbeddingAutoencoder, SentenceEmbeddingAutoencoderLoRA
from data_and_training import (
    load_corpus,
    prepare_dataloaders,
    train,
    setup_distributed_training
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Try importing wandb
try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False
    wandb = None

# Try importing deepspeed
try:
    import deepspeed
    _deepspeed_available = True
except ImportError:
    _deepspeed_available = False
    deepspeed = None


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # Additional settings for deterministic behavior
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a sentence embedding autoencoder")

    # --- Model ---
    g_model = parser.add_argument_group('Model Configuration')
    g_model.add_argument("--encoder_model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Encoder model name or path (Hugging Face)")
    g_model.add_argument("--decoder_model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Decoder model name or path (Hugging Face)")
    g_model.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization")
    g_model.add_argument("--embedding_dim", type=int, default=None, help="Target dimension of sentence embeddings (if projection needed, otherwise inferred)")
    g_model.add_argument("--pooling_strategy", type=str, default="last_token", choices=["mean", "cls", "last_token"], help="Pooling strategy")
    g_model.add_argument("--noise_std", type=float, default=0.1, help="Std deviation of Gaussian noise for blank tokens")
    g_model.add_argument("--noise_strategy", type=str, default="gaussian", choices=["gaussian", "dropout", "zeros"], help="Noise strategy")

    # --- LoRA ---
    g_lora = parser.add_argument_group('LoRA Configuration')
    g_lora.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    g_lora.add_argument("--lora_r", type=int, default=8, help="LoRA r parameter (rank)")
    g_lora.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter (scaling)")
    g_lora.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
    g_lora.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "v_proj"], help="Target modules for LoRA (e.g., 'q_proj' 'v_proj')")

    # --- Data ---
    g_data = parser.add_argument_group('Data Configuration')
    g_data.add_argument("--corpus", type=str, default="c4", help="Corpus name from Hugging Face datasets (e.g., c4, wikipedia, bookcorpus, openwebtext) or 'custom'")
    g_data.add_argument("--custom_corpus_path", type=str, default=None, help="Path to custom corpus file (.txt or .csv with 'text' column) if --corpus=custom")
    g_data.add_argument("--text_column", type=str, default="text", help="Name of the text column in the dataset")
    g_data.add_argument("--split", type=str, default="train", help="Data split to use (e.g., 'train', 'validation')")
    g_data.add_argument("--subset", type=str, default="en", help="Subset of the corpus (e.g., 'en' for C4). Default: 'en'.")
    g_data.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (None for all)")
    g_data.add_argument("--cache_dir", type=str, default=None, help="Directory for caching datasets")
    g_data.add_argument("--train_ratio", type=float, default=0.98, help="Ratio of data for training (vs validation)")
    g_data.add_argument("--streaming", action="store_true", help="Load dataset in streaming mode (for very large datasets)")

    # --- Training ---
    g_train = parser.add_argument_group('Training Configuration')
    g_train.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model checkpoints and logs")
    g_train.add_argument("--batch_size", type=int, default=8, help="Batch size PER GPU/PROCESS")
    g_train.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients before optimizer step")
    g_train.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate")
    g_train.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    g_train.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    g_train.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total steps for learning rate warmup")
    g_train.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant"], help="Learning rate scheduler type")
    g_train.add_argument("--num_workers", type=int, default=4, help="Number of dataloader worker processes")
    g_train.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    g_train.add_argument("--logging_steps", type=int, default=100, help="Log training status every N optimizer steps")
    g_train.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    g_train.add_argument("--fp16", action="store_true", help="Use native mixed precision training (not needed if using DeepSpeed FP16 config)")

    # --- Distributed Training ---
    g_dist = parser.add_argument_group('Distributed Training')
    # --distributed flag is handled by the launch script (torch.distributed.launch or deepspeed)
    # local_rank is automatically passed by the launcher
    g_dist.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (set by launcher)")
    g_dist.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed for training optimization")
    g_dist.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed configuration file (JSON)")

    # --- Logging ---
    g_log = parser.add_argument_group('Logging')
    g_log.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    g_log.add_argument("--wandb_project", type=str, default="sentence-embedding-autoencoder", help="W&B project name")
    g_log.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (defaults to timestamp)")

    return parser.parse_args()

def main():
    """Main training orchestration function."""
    args = parse_args()
    set_seed(args.seed)

    # --- Setup Distributed Training ---
    # setup_distributed_training figures out if it's distributed based on env vars
    # It initializes the process group if needed.
    device, is_distributed, rank, world_size, local_rank = setup_distributed_training(
        use_distributed=True, # Assume potentially distributed, let function check env vars
        local_rank=args.local_rank # Pass local rank from args (set by launcher)
    )
    args.local_rank = local_rank # Update args with the potentially discovered local_rank
    args.is_distributed = is_distributed
    args.world_size = world_size
    args.rank = rank

    # --- Create Output Directory (Rank 0 only) ---
    output_dir = args.output_dir
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"train_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        # Save arguments
        try:
            with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            logger.info(f"Saved training arguments to {os.path.join(output_dir, 'training_args.json')}")
        except Exception as e:
            logger.error(f"Failed to save training arguments: {e}")
    # Ensure all processes have the final output_dir path if needed later (e.g., for loading)
    if is_distributed:
         # Broadcast output_dir from rank 0 to all other ranks
         output_dir_list = [output_dir] if rank == 0 else [None]
         torch.distributed.broadcast_object_list(output_dir_list, src=0)
         output_dir = output_dir_list[0]
         args.output_dir = output_dir # Ensure all processes have the correct final path
         torch.distributed.barrier() # Ensure rank 0 created dir before others proceed


    # --- Initialize WandB (Rank 0 only) ---
    if args.wandb and rank == 0:
        if not _wandb_available:
            logger.warning("WandB logging requested but wandb library not found. Skipping.")
            args.wandb = False # Disable wandb if not installed
        else:
            wandb_run_name = args.wandb_run_name or os.path.basename(output_dir) # Use folder name if not specified
            try:
                 wandb.init(
                     project=args.wandb_project,
                     name=wandb_run_name,
                     config=vars(args),
                     dir=output_dir # Store wandb files within the output directory
                 )
                 logger.info(f"Initialized WandB logging for run: {wandb_run_name}")
            except Exception as e:
                 logger.error(f"Failed to initialize WandB: {e}")
                 args.wandb = False # Disable if init fails

    # --- Load Data ---
    logger.info(f"Loading corpus: {args.corpus}")
    if args.corpus == "custom":
        if not args.custom_corpus_path:
            raise ValueError("--custom_corpus_path must be provided when --corpus=custom")
        logger.info(f"Loading custom corpus from: {args.custom_corpus_path}")
        try:
            if args.custom_corpus_path.endswith(".txt"):
                with open(args.custom_corpus_path, "r", encoding="utf-8") as f:
                    texts = [line.strip() for line in f if line.strip()]
            elif args.custom_corpus_path.endswith(".csv"):
                 import pandas as pd
                 df = pd.read_csv(args.custom_corpus_path)
                 if args.text_column not in df.columns:
                      raise ValueError(f"Text column '{args.text_column}' not found in {args.custom_corpus_path}")
                 texts = df[args.text_column].dropna().astype(str).tolist()
            else:
                 raise ValueError(f"Unsupported custom corpus file format: {args.custom_corpus_path}. Use .txt or .csv")
            data_source = texts # List of strings
            args.streaming = False # Force non-streaming for custom files
        except Exception as e:
             logger.error(f"Failed to load custom corpus: {e}")
             sys.exit(1)
    else:
        # Load from Hugging Face datasets
        data_source = load_corpus(
            corpus_name=args.corpus,
            split=args.split,
            subset=args.subset,
            num_samples=args.num_samples,
            cache_dir=args.cache_dir,
            text_column=args.text_column,
            streaming=args.streaming
        )
        # If streaming, data_source is (dataset, text_column_name)
        # If not streaming, data_source is a list of texts

    if args.streaming:
         logger.info(f"Using streaming dataset.")
         # Need to handle the dataset object and text column in prepare_dataloaders/TextDataset
         # For now, assume prepare_dataloaders and TextDataset handle HF Dataset objects
         hf_dataset, _ = data_source # Unpack, we use args.text_column later if needed
         data_for_loader = hf_dataset
    else:
         logger.info(f"Loaded {len(data_source)} texts into memory.")
         data_for_loader = data_source


    # --- Initialize Model ---
    logger.info(f"Initializing model (Encoder: {args.encoder_model_name}, Decoder: {args.decoder_model_name})")
    model_class = SentenceEmbeddingAutoencoderLoRA if args.use_lora else SentenceEmbeddingAutoencoder
    model_kwargs = {
        "encoder_model_name": args.encoder_model_name,
        "decoder_model_name": args.decoder_model_name,
        "max_length": args.max_length,
        # "embedding_dim": args.embedding_dim, # Let model infer from decoder
        "pooling_strategy": args.pooling_strategy,
        "noise_std": args.noise_std,
        "noise_strategy": args.noise_strategy,
    }
    if args.use_lora:
        model_kwargs.update({
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
        })
        logger.info(f"Using LoRA with R={args.lora_r}, Alpha={args.lora_alpha}, Targets={args.lora_target_modules}")

    try:
         model = model_class(**model_kwargs)
    except Exception as e:
         logger.error(f"Failed to initialize model: {e}")
         # If OOM during init, suggest smaller model or loading with `device_map='auto'` if applicable
         if "out of memory" in str(e).lower():
              logger.error("Potential OOM during model loading. Consider a smaller base model, using LoRA, or DeepSpeed ZeRO stage 3.")
         sys.exit(1)

    # --- Prepare Tokenizer (shared for dataloader) ---
    # Assuming encoder and decoder tokenizers are compatible enough for input processing
    # Or use encoder_tokenizer specifically if that's the intended input format
    tokenizer = model.encoder_tokenizer


    # --- Prepare Dataloaders ---
    # Need to handle streaming dataset case if args.streaming is True
    from torch.utils.data import DataLoader
    from data_and_training import TextDataset
    
    if args.streaming:
         # For streaming, we pass the dataset object directly
         # TextDataset needs to be adapted or we need a different approach
         # Simple adaptation: Pass the dataset, TextDataset __getitem__ reads from it
         logger.info("Preparing dataloaders for streaming dataset...")
         # NOTE: Streaming datasets usually cannot be easily split into train/val like this.
         # You typically stream the 'train' split and use a separate 'validation' split if available.
         # Let's assume `data_for_loader` is the 'train' streaming dataset for now.
         # We might skip validation or load a separate small validation set non-streamed.
         train_dataset = TextDataset(data_for_loader, tokenizer, args.max_length)
         # Create a dummy or load a separate validation set
         # For simplicity, let's skip validation if streaming for now
         logger.warning("Validation skipped during streaming dataset training in this example.")
         val_loader = None

         train_sampler = None # Cannot easily use DistributedSampler with streaming datasets without IterableDataset adaptation
         if is_distributed:
             logger.warning("Distributed training with default streaming dataset may lead to data duplication. Consider using IterableDataset with worker-based sharding.")

         train_loader = DataLoader(
             train_dataset,
             batch_size=args.batch_size,
             num_workers=args.num_workers,
             pin_memory=True,
             sampler=train_sampler # Likely None
             # collate_fn might be needed depending on TextDataset output
         )
         num_training_steps_per_epoch = -1 # Unknown for streaming
         logger.info("Prepared streaming train dataloader (validation skipped).")

    else:
         # Non-streaming case (list of texts or pre-loaded HF Dataset)
         logger.info("Preparing dataloaders for in-memory data...")
         train_loader, val_loader = prepare_dataloaders(
             data=data_for_loader,
             tokenizer=tokenizer,
             max_length=args.max_length,
             train_ratio=args.train_ratio,
             batch_size=args.batch_size,
             num_workers=args.num_workers,
             is_distributed=is_distributed,
             seed=args.seed
         )
         num_training_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
         logger.info(f"Prepared dataloaders: ~{num_training_steps_per_epoch} training steps per epoch.")


    # --- Calculate Total Steps and Warmup Steps ---
    # Estimate total steps if possible (required for scheduler)
    if num_training_steps_per_epoch > 0:
         total_training_steps = num_training_steps_per_epoch * args.num_epochs
         num_warmup_steps = int(total_training_steps * args.warmup_ratio)
         logger.info(f"Total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")
    else:
         # Handle streaming case where length is unknown beforehand
         # Might need to set a very large number for total_steps or use a different scheduler
         logger.warning("Cannot determine exact total training steps for streaming dataset. Using estimated large number for scheduler.")
         # Estimate based on num_samples if provided, otherwise guess
         estimated_steps_per_epoch = (args.num_samples // (args.batch_size * args.world_size * args.gradient_accumulation_steps)) if args.num_samples else 100000 # Large guess
         total_training_steps = estimated_steps_per_epoch * args.num_epochs
         num_warmup_steps = int(total_training_steps * args.warmup_ratio)
         logger.info(f"Estimated total training steps: {total_training_steps}, Warmup steps: {num_warmup_steps}")


    # --- Prepare Optimizer, Scheduler, and DeepSpeed ---
    optimizer = None
    scheduler = None
    model_parameters = model.parameters()

    if args.use_deepspeed:
        if not _deepspeed_available:
            raise ImportError("DeepSpeed requested but not installed. Please install deepspeed.")

        logger.info("Initializing DeepSpeed...")
        # Load DeepSpeed config file if provided
        ds_config = None
        if args.deepspeed_config:
            try:
                with open(args.deepspeed_config, "r") as f:
                    ds_config = json.load(f)
                logger.info(f"Loaded DeepSpeed config from: {args.deepspeed_config}")
            except Exception as e:
                logger.error(f"Failed to load DeepSpeed config: {e}. Proceeding with default or command-line args.")
                ds_config = None # Fallback

        # If no config file, construct config from args (basic example)
        if ds_config is None:
             ds_config = {
                 "train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                 "optimizer": {
                     "type": "AdamW",
                     "params": {
                         "lr": args.learning_rate,
                         "weight_decay": args.weight_decay
                     }
                 },
                 "scheduler": {
                     "type": "WarmupLR", # Simple warmup, consider WarmupDecayLR
                     "params": {
                         "warmup_min_lr": 0,
                         "warmup_max_lr": args.learning_rate,
                         "warmup_num_steps": num_warmup_steps
                     }
                 },
                 "fp16": {"enabled": args.fp16}, # Control FP16 via deepspeed config
                 # Add ZeRO config here if needed, e.g., stage 2
                 "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                 }
             }
             logger.info("Using default DeepSpeed config generated from command-line args.")

        # Ensure scheduler params match estimates
        if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
             if ds_config["scheduler"]["type"] in ["WarmupLR", "WarmupDecayLR"]:
                 ds_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
             if ds_config["scheduler"]["type"] == "WarmupDecayLR":
                 ds_config["scheduler"]["params"]["total_num_steps"] = total_training_steps


        # Initialize DeepSpeed engine
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            config_params=ds_config
        )
        logger.info("DeepSpeed initialized successfully.")
        # DeepSpeed handles device placement, DDP wrapping, and FP16
        device = model_engine.local_rank # Use device from DeepSpeed engine
        model = model_engine # Use the DeepSpeed engine as the model


    else:
        # Standard PyTorch Optimizer and Scheduler
        logger.info("Using standard PyTorch optimizer and scheduler.")
        model.to(device) # Move model to device

        optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup( # Or other scheduler type based on args.lr_scheduler_type
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps
        )

        # Wrap model with DDP if distributed
        if is_distributed:
            logger.info(f"Wrapping model with DistributedDataParallel on rank {rank}.")
            # find_unused_parameters might be needed depending on model structure, esp. with LoRA
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == 'cuda' else None,
                output_device=local_rank if device.type == 'cuda' else None,
                find_unused_parameters=args.use_lora # Often needed for LoRA
            )

    # --- Start Training ---
    logger.info("***** Starting Training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, grad accum) = {args.batch_size * args.world_size * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Estimated total optimization steps = {total_training_steps}")
    logger.info(f"  Using {device.type.upper()} device: {device}")
    logger.info(f"  Mixed precision training: {'Enabled via DeepSpeed' if args.use_deepspeed and ds_config.get('fp16', {}).get('enabled') else 'Enabled via Native AMP' if args.fp16 and not args.use_deepspeed else 'Disabled'}")


    # Pass model config to train function for saving checkpoints
    from transformers import PretrainedConfig
    model_config_to_save = model.module.config if hasattr(model, 'module') else model.config
    if isinstance(model_config_to_save, PretrainedConfig): # Handle HF config object
         model_config_to_save = model_config_to_save.to_dict()


    train(
        model=model, # This is the DeepSpeed engine or DDP-wrapped model
        train_loader=train_loader,
        val_loader=val_loader, # Pass None if skipping validation
        optimizer=optimizer, # Managed by DeepSpeed if use_deepspeed=True
        scheduler=scheduler, # Managed by DeepSpeed if use_deepspeed=True
        device=device, # The device for the current process
        num_epochs=args.num_epochs,
        checkpoint_dir=output_dir, # Save checkpoints inside the run's output dir
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_every=args.save_every,
        use_deepspeed=args.use_deepspeed,
        fp16=(args.fp16 and not args.use_deepspeed), # Use native AMP only if not using DeepSpeed
        logging_steps=args.logging_steps,
        model_config=model_config_to_save # Pass the config dict
    )

    logger.info("Training finished.")

    # --- Clean up ---
    if args.wandb and rank == 0 and _wandb_available and wandb.run:
        wandb.finish()
        logger.info("WandB run finished.")

    if is_distributed:
        torch.distributed.destroy_process_group()
        logger.info("Distributed process group destroyed.")

if __name__ == "__main__":
    main()