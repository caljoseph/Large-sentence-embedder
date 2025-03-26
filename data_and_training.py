# data_and_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

# Try importing wandb, fail gracefully if not installed
try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False
    wandb = None # Define as None

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset for sentence embedding training."""

    def __init__(self, texts_or_dataset, tokenizer, max_length=128):
        """
        Args:
            texts_or_dataset: A list of strings or a Hugging Face Dataset object.
            tokenizer: The tokenizer to use.
            max_length: Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        if isinstance(texts_or_dataset, list):
            self.dataset = HFDataset.from_dict({"text": texts_or_dataset})
        elif isinstance(texts_or_dataset, HFDataset):
             self.dataset = texts_or_dataset
        else:
            raise TypeError("Input must be a list of strings or a datasets.Dataset object")

        # Pre-tokenize if possible and beneficial (can speed up loading)
        # This depends on dataset size and memory. For very large datasets,
        # tokenizing on-the-fly in __getitem__ might be better.
        # Let's stick to on-the-fly for simplicity now.

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Assumes the dataset has a 'text' column
        text = self.dataset[idx]['text']
        if not isinstance(text, str):
             text = str(text) # Ensure text is string

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length", # Pad to max_length consistently
            return_tensors="pt"
        )

        # Remove the batch dimension added by return_tensors="pt"
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Target is the same as input for autoencoder
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": input_ids.clone(),
            "target_mask": attention_mask.clone()
        }

def load_corpus(corpus_name="c4", split="train", subset=None, num_samples=None, cache_dir=None, text_column="text", streaming=False):
    """Load a text corpus from Hugging Face datasets."""
    logger.info(f"Loading dataset: {corpus_name}, subset: {subset}, split: {split}, streaming: {streaming}")
    try:
        if corpus_name == "c4":
            # Load the C4 dataset (Common Crawl)
            dataset = load_dataset("c4", subset or "en", split=split, cache_dir=cache_dir, streaming=streaming)
            text_column = "text"
        elif corpus_name == "wikipedia":
            # Load Wikipedia dataset
            dataset = load_dataset("wikipedia", "20220301.en", split=split, cache_dir=cache_dir, streaming=streaming)
            text_column = "text"
        elif corpus_name == "bookcorpus":
            # Load BookCorpus dataset
            dataset = load_dataset("bookcorpus", split=split, cache_dir=cache_dir, streaming=streaming)
            text_column = "text"
        elif corpus_name == "openwebtext":
            dataset = load_dataset("openwebtext", split=split, cache_dir=cache_dir, streaming=streaming)
            text_column = "text"
        # Add more standard datasets here
        else:
             # Try loading directly by name, assuming standard format
             logger.warning(f"Attempting to load '{corpus_name}' directly. Assuming '{text_column}' column.")
             dataset = load_dataset(corpus_name, subset, split=split, cache_dir=cache_dir, streaming=streaming)

    except Exception as e:
        raise ValueError(f"Failed to load dataset '{corpus_name}': {e}")

    # Limit the number of samples if specified AND not streaming
    if not streaming and num_samples is not None and num_samples > 0:
         if num_samples < len(dataset):
              logger.info(f"Selecting first {num_samples} samples from the dataset.")
              dataset = dataset.shuffle(seed=42).select(range(num_samples))
         else:
              logger.warning(f"num_samples ({num_samples}) >= dataset size ({len(dataset)}). Using full dataset.")
    elif streaming and num_samples is not None and num_samples > 0:
         logger.info(f"Taking first {num_samples} samples from the streaming dataset.")
         dataset = dataset.take(num_samples)


    # If streaming, we return the dataset object directly
    if streaming:
        logger.info(f"Returning streaming dataset object for '{corpus_name}'. Text column: '{text_column}'.")
        # Note: Need to handle text extraction within the Dataset/DataLoader later
        return dataset, text_column # Return dataset and the column name

    # If not streaming, extract texts
    logger.info(f"Extracting '{text_column}' from the loaded dataset.")
    try:
        texts = dataset[text_column]
        return texts
    except KeyError:
        raise ValueError(f"Text column '{text_column}' not found in dataset '{corpus_name}'. Available columns: {dataset.column_names}")
    except Exception as e:
         raise RuntimeError(f"Error extracting text from non-streaming dataset: {e}")


def prepare_dataloaders(
    data, # Can be list of texts or HF Dataset
    tokenizer,
    max_length=128,
    train_ratio=0.95,
    batch_size=32,
    num_workers=4,
    is_distributed=False,
    seed=42
):
    """Prepare training and validation dataloaders."""

    if isinstance(data, list):
        # Split list into train and validation
        np.random.seed(seed)
        np.random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        train_texts = data[:split_idx]
        val_texts = data[split_idx:]

        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")

        # Create datasets
        train_dataset = TextDataset(train_texts, tokenizer, max_length)
        val_dataset = TextDataset(val_texts, tokenizer, max_length)

    elif isinstance(data, HFDataset):
         # Split HF Dataset
         logger.info(f"Splitting Hugging Face Dataset (size: {len(data)}) with train_ratio: {train_ratio}")
         # Ensure dataset is not too small for splitting
         if len(data) < 2:
             raise ValueError("Dataset is too small to be split.")

         # Use train_test_split for robustness
         split_data = data.train_test_split(test_size=1.0 - train_ratio, seed=seed)
         train_hf_dataset = split_data["train"]
         val_hf_dataset = split_data["test"]

         logger.info(f"Training samples: {len(train_hf_dataset)}")
         logger.info(f"Validation samples: {len(val_hf_dataset)}")

         train_dataset = TextDataset(train_hf_dataset, tokenizer, max_length)
         val_dataset = TextDataset(val_hf_dataset, tokenizer, max_length)
    else:
        raise TypeError("Input 'data' must be a list of strings or a datasets.Dataset object")


    # Create samplers for distributed training if needed
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None), # Shuffle only if not using distributed sampler
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler
    )

    return train_loader, val_loader

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, gradient_accumulation_steps=1, use_deepspeed=False, scaler=None, logging_steps=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_steps = 0
    total_samples = 0
    epoch_start_time = time.time()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", disable=(use_deepspeed and torch.distributed.get_rank() != 0))

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Forward pass
        if scaler: # Native AMP
             with torch.cuda.amp.autocast():
                 outputs = model(**batch)
                 loss = outputs["loss"]
        else: # No AMP or DeepSpeed handles it
             outputs = model(**batch)
             loss = outputs["loss"]

        # Handle potential loss reduction in DDP
        if not use_deepspeed and hasattr(model, 'module') and loss is not None:
            # If using DDP, loss might be averaged across GPUs already.
            # If not, ensure it's a scalar. Usually, HF models return averaged loss.
            pass # Assuming loss is already averaged

        if loss is None:
             logger.warning(f"Step {step}: Loss is None. Skipping backward pass.")
             continue

        # Scale loss for gradient accumulation
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        # Backward pass
        if use_deepspeed:
            model.backward(loss)
        elif scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_deepspeed:
                model.step() # DeepSpeed handles optimizer, scheduler, and gradient clipping
            else:
                if scaler:
                     scaler.unscale_(optimizer) # Unscale gradients before clipping
                     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Optional gradient clipping
                     scaler.step(optimizer)
                     scaler.update()
                else:
                     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Optional gradient clipping
                     optimizer.step()

                if scheduler: # Step scheduler for non-DeepSpeed setups
                     scheduler.step()
                optimizer.zero_grad()

            num_steps += 1

            # Track statistics
            current_loss = loss.item() * gradient_accumulation_steps # Get unscaled loss
            total_loss += current_loss
            avg_loss_step = total_loss / num_steps

            # Log to console (rank 0 only)
            if not use_deepspeed or torch.distributed.get_rank() == 0:
                if num_steps % logging_steps == 0:
                    lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                    elapsed_time = time.time() - epoch_start_time
                    logger.info(f"Epoch {epoch} | Step {num_steps}/{len(dataloader)//gradient_accumulation_steps} | "
                                f"Loss: {current_loss:.4f} | Avg Loss: {avg_loss_step:.4f} | LR: {lr:.2e} | Time: {elapsed_time:.2f}s")

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{avg_loss_step:.4f}", "lr": f"{lr:.2e}"})

                # Log to wandb (rank 0 only)
                if _wandb_available and wandb.run:
                    wandb.log({
                        "train/step_loss": current_loss,
                        "train/learning_rate": lr,
                        "train/epoch": epoch + (step / len(dataloader)), # Fractional epoch
                        "global_step": num_steps # Assuming global step counter needed
                    })


        total_samples += batch["input_ids"].size(0) # * world_size if using DDP? Check this.

    # Final average loss for the epoch
    avg_epoch_loss = total_loss / num_steps if num_steps > 0 else 0
    epoch_duration = time.time() - epoch_start_time
    logger.info(f"Epoch {epoch} Training finished. Average Loss: {avg_epoch_loss:.4f}. Duration: {epoch_duration:.2f}s")
    return avg_epoch_loss


def evaluate(model, dataloader, device, epoch, use_deepspeed=False):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_steps = 0
    eval_start_time = time.time()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Eval]", disable=(use_deepspeed and torch.distributed.get_rank() != 0))

    with torch.no_grad():
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Forward pass
            # No need for autocast/scaler during evaluation if using FP32 evaluation
            # If evaluating in FP16 (less common), would need autocast
            outputs = model(**batch)
            loss = outputs["loss"]

            if loss is not None:
                # Handle potential reduction in DDP for metrics gathering if needed
                # For simple loss averaging, just sum and divide
                total_loss += loss.item()
                num_steps += 1

                # Update progress bar (rank 0 only)
                if not use_deepspeed or torch.distributed.get_rank() == 0:
                    avg_loss = total_loss / num_steps
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    avg_loss = total_loss / num_steps if num_steps > 0 else 0
    eval_duration = time.time() - eval_start_time

    # Aggregate loss across devices if using DDP/DeepSpeed
    if use_deepspeed or (hasattr(torch, 'distributed') and torch.distributed.is_initialized()):
         avg_loss_tensor = torch.tensor(avg_loss, device=device)
         torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.AVG)
         avg_loss = avg_loss_tensor.item()

    logger.info(f"Epoch {epoch} Evaluation finished. Average Loss: {avg_loss:.4f}. Duration: {eval_duration:.2f}s")
    return avg_loss

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_dir,
    gradient_accumulation_steps=1,
    save_every=1,
    use_deepspeed=False,
    fp16=False, # Native fp16 flag
    logging_steps=100,
    model_config=None # Pass model config for saving
):
    """Full training loop."""
    best_val_loss = float("inf")
    start_epoch = 1

    # Setup native AMP GradScaler if fp16 enabled and not using DeepSpeed
    scaler = None
    if fp16 and not use_deepspeed:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("Using native PyTorch AMP with GradScaler.")

    # --- Checkpoint Loading (Basic Example) ---
    # More robust checkpointing would handle optimizer, scheduler, epoch, etc.
    # resume_from_checkpoint = None # Set path to resume
    # if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
    #     logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    #     checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')
    #     if use_deepspeed:
    #         # DeepSpeed handles loading its checkpoint format
    #         model.load_checkpoint(resume_from_checkpoint)
    #     else:
    #         # Basic state dict loading
    #         model_to_load = model.module if hasattr(model, 'module') else model
    #         model_to_load.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         if scheduler and 'scheduler_state_dict' in checkpoint:
    #             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #         if scaler and 'scaler_state_dict' in checkpoint:
    #              scaler.load_state_dict(checkpoint['scaler_state_dict'])
    #     start_epoch = checkpoint.get('epoch', 1) + 1
    #     best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    #     logger.info(f"Resumed from epoch {start_epoch-1}. Best val loss so far: {best_val_loss:.4f}")


    # Determine if we should save checkpoints (only rank 0 in distributed setting)
    should_save = not use_deepspeed or torch.distributed.get_rank() == 0
    if should_save and checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    elif should_save and not checkpoint_dir:
        logger.warning("Checkpoint directory not specified. Checkpoints will not be saved.")


    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{num_epochs}")

        # Set epoch for distributed sampler if used
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, 'set_epoch'):
             val_loader.sampler.set_epoch(epoch)


        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_deepspeed=use_deepspeed,
            scaler=scaler,
            logging_steps=logging_steps
        )

        # Evaluate
        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            use_deepspeed=use_deepspeed
        )

        # Log metrics (rank 0 only)
        if not use_deepspeed or torch.distributed.get_rank() == 0:
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            if _wandb_available and wandb.run:
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss
                })

            # Save checkpoint if it's the best model so far or at save interval
            if checkpoint_dir:
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")

                if is_best or (epoch % save_every == 0):
                    save_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                    best_path = os.path.join(checkpoint_dir, "best_model.pt")

                    if use_deepspeed:
                        # DeepSpeed handles saving model, optimizer, scheduler state
                        # Tagging allows saving specific checkpoints
                        save_tag = f"epoch_{epoch}"
                        model.save_checkpoint(checkpoint_dir, tag=save_tag)
                        logger.info(f"Saved DeepSpeed checkpoint tagged '{save_tag}' to {checkpoint_dir}")
                        # Optionally, save a reference or link for 'best_model' if needed,
                        # or handle best model logic based on DeepSpeed's checkpoint structure.
                        if is_best:
                             # DeepSpeed doesn't have a direct 'save best' concept like simple saving.
                             # You might copy the best checkpoint folder or manage it externally.
                             logger.info(f"Best DeepSpeed checkpoint is tagged '{save_tag}' (manual tracking required).")

                    else:
                        # Standard PyTorch saving
                        model_to_save = model.module if hasattr(model, 'module') else model
                        save_obj = {
                            "epoch": epoch,
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "best_val_loss": best_val_loss,
                            "model_config": model_config # Save model config
                        }
                        if scheduler:
                            save_obj["scheduler_state_dict"] = scheduler.state_dict()
                        if scaler:
                            save_obj["scaler_state_dict"] = scaler.state_dict()

                        torch.save(save_obj, save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                        if is_best:
                            torch.save(save_obj, best_path)
                            logger.info(f"Saved best model checkpoint to {best_path}")


    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return best_val_loss

def setup_distributed_training(use_distributed=True, local_rank=-1, backend='nccl'):
    """Set up distributed training environment."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", local_rank)) # Get local rank from env if available

    is_distributed = use_distributed and world_size > 1

    if is_distributed:
        if local_rank == -1:
             raise ValueError("local_rank must be provided or set in environment for distributed training.")

        # Set the device for the current process
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # Initialize the process group
        if not torch.distributed.is_initialized():
            logger.info(f"Initializing distributed process group (backend: {backend}) | Rank: {rank}/{world_size} | Local Rank: {local_rank}")
            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
        else:
             logger.warning("Distributed process group already initialized.")

        # Synchronize before starting training
        torch.distributed.barrier()
        logger.info("Distributed environment initialized.")

    else:
        # Single process or CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Running in non-distributed mode.")

    # Ensure reproducibility if desired (needs to be done on all processes)
    # seed = 42
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    return device, is_distributed, rank, world_size, local_rank