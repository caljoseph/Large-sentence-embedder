#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a trained sentence embedding model on various benchmarks.

This script loads a trained SentenceEmbeddingAutoencoder model (or a baseline)
and evaluates it on a suite of benchmarks including STS, classification, clustering.
"""

import os
import sys
import torch
import logging
import argparse
from datetime import datetime
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import project modules AFTER setting path
from model.sentence_embedding_model import SentenceEmbeddingAutoencoder, SentenceEmbeddingAutoencoderLoRA
from evaluation_suite import SentenceEmbeddingEvaluator, compare_models

# Try importing SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    _st_available = True
except ImportError:
    _st_available = False
    SentenceTransformer = None

# Try importing wandb
try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False
    wandb = None

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained sentence embedding model")

    # --- Model Loading ---
    g_model = parser.add_argument_group('Model Loading')
    g_model.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pt file) or directory (for PEFT/full model save).")
    g_model.add_argument("--model_name", type=str, default="our_trained_model",
                        help="Name for the model being evaluated (used in logs/results).")
    # Args needed if loading from state dict only and config isn't saved in checkpoint
    g_model.add_argument("--encoder_model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base encoder model name (if loading state dict only).")
    g_model.add_argument("--decoder_model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base decoder model name (if loading state dict only).")
    g_model.add_argument("--use_lora", action="store_true",
                        help="Specify if the loaded model uses LoRA adapters.")
    g_model.add_argument("--pooling_strategy", type=str, default=None,
                         help="Override pooling strategy if not in config or checkpoint.")


    # --- Evaluation Configuration ---
    g_eval = parser.add_argument_group('Evaluation Configuration')
    g_eval.add_argument("--tasks", type=str, nargs="+", default=["sts", "classification", "clustering"],
                        choices=["sts", "classification", "clustering", "retrieval", "all"],
                        help="List of evaluation task types to run.")
    g_eval.add_argument("--compare_with", type=str, nargs="+", default=[],
                        help="List of baseline model names from sentence-transformers to compare against.")
    g_eval.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding during evaluation.")
    g_eval.add_argument("--max_samples_cluster", type=int, default=2000,
                        help="Maximum number of samples for clustering tasks.")
    g_eval.add_argument("--max_train_samples_classif", type=int, default=10000,
                        help="Maximum number of training samples for classification probes.")
    g_eval.add_argument("--max_eval_samples_classif", type=int, default=None,
                         help="Maximum number of evaluation samples for classification probes.")
    g_eval.add_argument("--dataset_cache_dir", type=str, default=None,
                        help="Directory for caching downloaded datasets.")
    g_eval.add_argument("--skip_visualization", action="store_true",
                         help="Skip generating t-SNE/UMAP visualizations.")
    g_eval.add_argument("--vis_dataset", type=str, default="sst2",
                         help="Dataset to use for embedding visualization.")
    g_eval.add_argument("--vis_max_samples", type=int, default=1000,
                         help="Max samples for visualization.")


    # --- Output & Logging ---
    g_out = parser.add_argument_group('Output & Logging')
    g_out.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results and plots.")
    g_out.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases for logging evaluation results.")
    g_out.add_argument("--wandb_project", type=str, default="sentence-embedding-eval",
                        help="W&B project name for evaluation.")
    g_out.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name for evaluation (defaults to model name + timestamp).")

    return parser.parse_args()

def load_trained_model(args):
     """ Loads the custom trained model from checkpoint or directory. """
     logger.info(f"Attempting to load model from: {args.model_path}")
     model = None
     model_config = {}

     # Check if path is a directory (potential save_pretrained format)
     if os.path.isdir(args.model_path):
         config_path = os.path.join(args.model_path, "config.json")
         if os.path.exists(config_path):
             logger.info("Detected directory structure, attempting to load using from_pretrained...")
             try:
                 # Determine if it's LoRA based on config or args
                 with open(config_path, 'r') as f:
                     loaded_config = json.load(f)
                 is_lora = args.use_lora or any(k.startswith("lora_") for k in loaded_config)

                 if is_lora:
                      logger.info("Loading as LoRA model.")
                      model = SentenceEmbeddingAutoencoderLoRA.from_pretrained(args.model_path)
                 else:
                      logger.info("Loading as standard model.")
                      model = SentenceEmbeddingAutoencoder.from_pretrained(args.model_path)
                 model_config = model.config # Use config loaded by the model
                 logger.info("Model loaded successfully using from_pretrained.")
             except Exception as e:
                 logger.error(f"Failed to load model using from_pretrained from directory {args.model_path}: {e}")
                 logger.info("Falling back to state dict loading (may require correct args).")
                 model = None # Reset model to trigger fallback
         else:
             logger.warning(f"Directory {args.model_path} exists but config.json not found. Assuming it's not a save_pretrained directory.")


     # Fallback or direct load from .pt file
     if model is None and os.path.isfile(args.model_path) and args.model_path.endswith(".pt"):
         logger.info(f"Loading model state_dict from checkpoint file: {args.model_path}")
         try:
             checkpoint = torch.load(args.model_path, map_location="cpu")

             # Try to load config from checkpoint
             if 'model_config' in checkpoint:
                 model_config = checkpoint['model_config']
                 logger.info("Loaded model configuration from checkpoint.")
             else:
                 logger.warning("Model configuration not found in checkpoint. Using command-line arguments for model structure.")
                 model_config = { # Construct from args
                     "encoder_model_name": args.encoder_model_name,
                     "decoder_model_name": args.decoder_model_name,
                     "max_length": 128, # Default, ideally load actual value
                     "embedding_dim": None,
                     "pooling_strategy": args.pooling_strategy or "last_token",
                     "noise_std": 0.1,
                     "noise_strategy": "gaussian",
                 }
                 if args.use_lora:
                     model_config.update({
                         "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, # Defaults
                     })

             # Override pooling strategy if specified
             if args.pooling_strategy:
                  model_config["pooling_strategy"] = args.pooling_strategy


             # Initialize model architecture based on config
             model_class = SentenceEmbeddingAutoencoderLoRA if args.use_lora else SentenceEmbeddingAutoencoder
             logger.info(f"Initializing model structure: {model_class.__name__}")
             model = model_class(**{k: v for k, v in model_config.items() if k not in ['lora_target_modules']}) # Filter non-init args if needed


             # Load state dict
             state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict" # Check common keys
             if state_dict_key in checkpoint:
                  # Handle potential DDP prefix 'module.'
                  state_dict = checkpoint[state_dict_key]
                  if all(k.startswith('module.') for k in state_dict):
                       logger.info("Removing 'module.' prefix from state dict keys.")
                       state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
                  # Handle PEFT loading if needed (might require specific loading logic for adapters)
                  # Basic load_state_dict works for full models or if PEFT adapters are part of the state_dict
                  model.load_state_dict(state_dict, strict=False) # Use strict=False initially for flexibility
                  logger.info("Model state_dict loaded successfully.")
             else:
                  logger.error(f"Could not find model state dict in checkpoint {args.model_path} (checked keys: 'model_state_dict', 'state_dict').")
                  return None, {}


         except Exception as e:
             logger.error(f"Failed to load model from checkpoint {args.model_path}: {e}")
             return None, {}
     elif model is None:
          logger.error(f"Model path not found or invalid: {args.model_path}")
          return None, {}


     return model, model_config


def main():
    """Main evaluation function."""
    args = parse_args()

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Create Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{args.model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {output_dir}")

    # Save evaluation arguments
    try:
        with open(os.path.join(output_dir, "evaluation_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    except Exception as e:
         logger.error(f"Failed to save evaluation arguments: {e}")

    # --- Initialize WandB ---
    if args.wandb:
        if not _wandb_available:
            logger.warning("WandB logging requested but wandb library not found. Skipping.")
            args.wandb = False
        else:
            wandb_run_name = args.wandb_run_name or f"eval_{args.model_name}_{timestamp}"
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=wandb_run_name,
                    config=vars(args),
                    dir=output_dir
                )
                logger.info(f"Initialized WandB logging for run: {wandb_run_name}")
            except Exception as e:
                logger.error(f"Failed to initialize WandB: {e}")
                args.wandb = False

    # --- Load Our Trained Model ---
    our_model, model_config = load_trained_model(args)
    if our_model is None:
        logger.error("Failed to load the specified model. Exiting.")
        sys.exit(1)

    our_model.to(device)
    our_model.eval()

    # --- Prepare Models Dictionary for Comparison ---
    models_to_evaluate = {args.model_name: our_model}

    # Load baseline models if requested
    if args.compare_with:
        if not _st_available:
             logger.warning("SentenceTransformer library not found. Cannot load baseline models.")
        else:
            for baseline_name in args.compare_with:
                logger.info(f"Loading baseline model: {baseline_name}")
                try:
                    baseline_model = SentenceTransformer(baseline_name, device=device, cache_folder=args.dataset_cache_dir)
                    baseline_model.eval() # Ensure eval mode
                    models_to_evaluate[baseline_name] = baseline_model
                    logger.info(f"Loaded baseline {baseline_name} successfully.")
                except Exception as e:
                    logger.error(f"Failed to load baseline model '{baseline_name}': {e}")


    # --- Run Evaluation Comparison ---
    logger.info("***** Starting Evaluation Comparison *****")
    logger.info(f"Models to evaluate: {list(models_to_evaluate.keys())}")
    logger.info(f"Tasks: {args.tasks}")

    # The compare_models function orchestrates evaluation and plotting
    all_results = compare_models(
        models_dict=models_to_evaluate,
        output_dir=output_dir,
        batch_size=args.batch_size,
        max_samples_cluster=args.max_samples_cluster,
        max_train_samples_classif=args.max_train_samples_classif,
        max_eval_samples_classif=args.max_eval_samples_classif,
        run_visualization=(not args.skip_visualization),
        vis_dataset=args.vis_dataset,
        vis_max_samples=args.vis_max_samples
        # Pass specific task lists if needed, though evaluate_all inside covers common ones
    )

    logger.info(f"***** Evaluation Finished *****")
    logger.info(f"Results saved in: {output_dir}")

    # --- Final WandB Logging ---
    if args.wandb and _wandb_available and wandb.run:
         try:
             # Log summary metrics if available
             summary = {}
             for model, results in all_results.items():
                  summary[f"{model}/sts_spearman_avg"] = results.get("sts", {}).get("average", {}).get("spearman", 0)
                  summary[f"{model}/classif_acc_avg"] = results.get("classification", {}).get("average", {}).get("accuracy", 0)
                  summary[f"{model}/cluster_nmi_avg"] = results.get("clustering", {}).get("average", {}).get("nmi", 0)
                  summary[f"{model}/overall_avg"] = results.get("overall_average", 0)
             wandb.log(summary)

             # Log results table (optional, can be large)
             # results_df = pd.DataFrame(all_results) # Process into suitable format
             # wandb.log({"evaluation_summary_table": wandb.Table(dataframe=results_df)})

             wandb.finish()
             logger.info("WandB run finished.")
         except Exception as e:
              logger.error(f"Failed during final WandB logging: {e}")


if __name__ == "__main__":
    main()