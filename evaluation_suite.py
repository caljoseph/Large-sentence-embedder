# evaluation_suite.py

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset as HFDataset
import logging
from typing import Dict, List, Optional, Tuple, Union
import pickle
import json
import time

# Try importing SentenceTransformer, fail gracefully
try:
    from sentence_transformers import SentenceTransformer
    _st_available = True
except ImportError:
    _st_available = False
    SentenceTransformer = None # Define as None

# Try importing wandb
try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False
    wandb = None


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SentenceEmbeddingEvaluator:
    """
    Evaluator for sentence embedding models on various benchmark tasks.
    Handles both custom models with an `encode` method and SentenceTransformer models.
    """

    def __init__(self,
                 model, # Must have an .encode() method
                 model_name_or_path=None, # Optional: For logging/identification
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 cache_dir=None):
        """
        Initialize the evaluator with a model.

        Args:
            model: A sentence embedding model instance with an encode() method
                   (e.g., our SentenceEmbeddingAutoencoder or a SentenceTransformer).
            model_name_or_path: A name or path for identifying the model in logs/results.
            device: Device to use for inference ('cuda' or 'cpu').
            cache_dir: Cache directory for datasets.
        """
        self.device = device
        self.cache_dir = cache_dir
        self.model_name = model_name_or_path or "custom_model"

        if not hasattr(model, 'encode'):
            raise TypeError("The provided model must have an 'encode' method.")

        self.model = model
        # Move model to device if it's a PyTorch module
        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)
        logger.info(f"Initialized evaluator for model '{self.model_name}' on device '{self.device}'")

    def _encode_batch(self, sentences: List[str], batch_size: int) -> np.ndarray:
        """ Encodes a list of sentences in batches and returns numpy array. """
        all_embeddings = []
        self.model.eval() # Ensure eval mode

        # Check if model is a SentenceTransformer for potential optimizations
        is_st_model = _st_available and isinstance(self.model, SentenceTransformer)

        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding", leave=False):
                batch = sentences[i:i+batch_size]
                if is_st_model:
                    # ST encode method handles batching and device internally
                    # convert_to_numpy=True is efficient
                    embeddings = self.model.encode(batch,
                                                   batch_size=len(batch), # Pass current batch size
                                                   device=self.device,
                                                   show_progress_bar=False,
                                                   convert_to_numpy=True)
                else:
                    # Assume custom model's encode method
                    # We handle device passing and batching externally
                    # Ensure model is on the correct device (done in __init__)
                    embeddings_tensor = self.model.encode(batch, batch_size=len(batch), device=self.device)
                    embeddings = embeddings_tensor.cpu().numpy() # Move to CPU and convert to numpy

                all_embeddings.append(embeddings)

        if not all_embeddings:
             return np.array([]) # Handle empty input case

        return np.concatenate(all_embeddings, axis=0)

    def evaluate_sts(self,
                     dataset_name="stsb_multi_mt",
                     split="test",
                     batch_size=32,
                     return_results=False):
        """
        Evaluate on Semantic Textual Similarity benchmark.

        Args:
            dataset_name: Name of the dataset (e.g., 'stsb_multi_mt', 'sickr', 'sts12' through 'sts16').
            split: Data split to use ('test' or 'validation').
            batch_size: Batch size for encoding.
            return_results: Whether to return detailed results alongside metrics.

        Returns:
            Dictionary with Pearson and Spearman correlation scores, optionally detailed results.
        """
        # --- Dataset Loading Logic ---
        logger.info(f"Loading STS dataset: {dataset_name}, split: {split}")
        try:
            if dataset_name == "stsb_multi_mt":
                dataset = load_dataset("stsb_multi_mt", "en", split=split, cache_dir=self.cache_dir)
                sentence1_key, sentence2_key, score_key, score_scale = "sentence1", "sentence2", "similarity_score", 5.0
            elif dataset_name == "sickr": # SICK-Relatedness
                dataset = load_dataset("sick", split=split, cache_dir=self.cache_dir) # Use 'validation' or 'test' split
                sentence1_key, sentence2_key, score_key, score_scale = "sentence_A", "sentence_B", "relatedness_score", 5.0
            elif dataset_name.startswith("sts") and dataset_name[3:].isdigit() and 12 <= int(dataset_name[3:]) <= 16:
                # Handles sts12, sts13, sts14, sts15, sts16
                # These are often available via specific loaders or MTEB helper
                try:
                     # Try MTEB format if available
                     dataset = load_dataset(f"mteb/{dataset_name}-sts", split=split, cache_dir=self.cache_dir)
                except Exception as e_mteb:
                     logger.warning(f"Could not load {dataset_name} via mteb ({e_mteb}). Trying direct load (may fail).")
                     # Fallback attempt - might need specific loading logic per year
                     dataset = load_dataset(dataset_name, split=split, cache_dir=self.cache_dir) # This might fail
                sentence1_key, sentence2_key, score_key, score_scale = "sentence1", "sentence2", "score", 5.0
            else:
                raise ValueError(f"Unsupported STS dataset: {dataset_name}")

        except Exception as e:
             logger.error(f"Failed to load or process dataset {dataset_name}: {e}")
             return {"pearson": 0.0, "spearman": 0.0} # Return default failure values

        # --- Data Extraction ---
        try:
            sentences1 = dataset[sentence1_key]
            sentences2 = dataset[sentence2_key]
            gold_scores = np.array(dataset[score_key])
            # Normalize scores to [0, 1] based on the dataset's scale
            gold_scores_norm = gold_scores / score_scale
        except KeyError as e:
             logger.error(f"Missing expected column in {dataset_name}: {e}. Columns: {dataset.column_names}")
             return {"pearson": 0.0, "spearman": 0.0}
        except Exception as e:
             logger.error(f"Error during data extraction for {dataset_name}: {e}")
             return {"pearson": 0.0, "spearman": 0.0}


        # --- Encoding ---
        logger.info(f"Encoding {len(sentences1)} sentence pairs for {dataset_name}...")
        start_time = time.time()
        embeddings1 = self._encode_batch(sentences1, batch_size)
        embeddings2 = self._encode_batch(sentences2, batch_size)
        encoding_time = time.time() - start_time
        logger.info(f"Encoding finished in {encoding_time:.2f} seconds.")

        if embeddings1.shape[0] == 0 or embeddings2.shape[0] == 0:
             logger.error(f"Encoding produced empty results for {dataset_name}.")
             return {"pearson": 0.0, "spearman": 0.0}

        # --- Calculate Cosine Similarities ---
        # Ensure embeddings are normalized for potentially faster cosine calculation
        # embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        # embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        # cosine_scores = np.sum(embeddings1 * embeddings2, axis=1) # Dot product of normalized vectors

        # Or use sklearn's pairwise (handles non-normalized too) - safer default
        cosine_scores = np.array([sk_cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0, 0]
                                 for e1, e2 in zip(embeddings1, embeddings2)])


        # --- Calculate Correlations ---
        try:
            pearson_corr, _ = pearsonr(gold_scores_norm, cosine_scores)
            spearman_corr, _ = spearmanr(gold_scores_norm, cosine_scores)
        except ValueError as e:
             logger.error(f"Correlation calculation failed for {dataset_name} (likely due to constant scores): {e}")
             pearson_corr, spearman_corr = 0.0, 0.0

        results = {
            "pearson": pearson_corr * 100,  # Convert to percentage
            "spearman": spearman_corr * 100  # Convert to percentage
        }

        logger.info(f"Results for {dataset_name}: Pearson={results['pearson']:.2f}%, Spearman={results['spearman']:.2f}%")

        if return_results:
            return results, {
                "sentences1": sentences1,
                "sentences2": sentences2,
                "gold_scores": gold_scores_norm.tolist(),
                "predicted_scores": cosine_scores.tolist(),
                "embeddings1": embeddings1, # Optional: return embeddings too
                "embeddings2": embeddings2,
            }

        return results

    def evaluate_sts_all(self, batch_size=32, datasets=None):
        """
        Evaluate on multiple STS benchmarks.

        Args:
            batch_size: Batch size for encoding.
            datasets: List of STS dataset names to evaluate. Defaults to a standard set.

        Returns:
            Dictionary with correlation scores for all specified STS datasets and an average.
        """
        if datasets is None:
            # Common STS benchmarks used in evaluations
            datasets = ["stsb_multi_mt", "sickr", "sts12", "sts13", "sts14", "sts15", "sts16"]

        all_results = {}
        valid_results_count = 0
        total_pearson = 0.0
        total_spearman = 0.0

        for dataset_name in datasets:
            try:
                # Determine appropriate split (usually 'test', but some might only have 'validation')
                # This logic might need refinement based on specific datasets
                split = 'test'
                if dataset_name in ["sickr"]: # Example: SICK often uses validation or test
                     split = 'test' # Or 'validation' depending on standard practice

                dataset_results = self.evaluate_sts(dataset_name=dataset_name, split=split, batch_size=batch_size)

                # Check if results are valid (not the default 0.0 failure values)
                # Use a small epsilon to avoid floating point issues if 0.0 is a possible valid score
                if abs(dataset_results["pearson"]) > 1e-6 or abs(dataset_results["spearman"]) > 1e-6:
                    all_results[dataset_name] = dataset_results
                    total_pearson += dataset_results["pearson"]
                    total_spearman += dataset_results["spearman"]
                    valid_results_count += 1
                else:
                    logger.warning(f"Skipping {dataset_name} results in average due to likely failure (scores are zero).")
                    all_results[dataset_name] = {"pearson": 0.0, "spearman": 0.0} # Still record failure

            except Exception as e:
                logger.error(f"Error evaluating {dataset_name}: {str(e)}")
                all_results[dataset_name] = {"pearson": 0.0, "spearman": 0.0}

        # Calculate average over datasets with valid results
        avg_pearson = (total_pearson / valid_results_count) if valid_results_count > 0 else 0.0
        avg_spearman = (total_spearman / valid_results_count) if valid_results_count > 0 else 0.0

        all_results["average"] = {
            "pearson": avg_pearson,
            "spearman": avg_spearman
        }
        logger.info(f"STS Average (over {valid_results_count} datasets): Pearson={avg_pearson:.2f}%, Spearman={avg_spearman:.2f}%")

        return all_results

    def evaluate_classification(self,
                               dataset_name="sst2",
                               split="validation", # Split for evaluation
                               train_split="train", # Split for training the classifier
                               batch_size=32,
                               max_train_samples=10000, # Limit classifier training data
                               max_eval_samples=None, # Limit evaluation data (optional)
                               text_key=None, # Specify text column if non-standard
                               label_key="label",
                               return_results=False):
        """
        Evaluate on sentence classification tasks using logistic regression trained on embeddings.

        Args:
            dataset_name: Name of the dataset (e.g., 'sst2', 'imdb', 'ag_news', 'trec').
            split: Data split to evaluate the trained classifier on.
            train_split: Data split to train the logistic regression classifier on.
            batch_size: Batch size for encoding sentences.
            max_train_samples: Max samples to use for training the classifier.
            max_eval_samples: Max samples to use for evaluating the classifier.
            text_key: Name of the column containing the text. Auto-detected for common datasets.
            label_key: Name of the column containing the labels.
            return_results: Whether to return detailed results.

        Returns:
            Dictionary with classification metrics (accuracy, F1), optionally detailed results.
        """
        # --- Dataset Loading and Key Identification ---
        logger.info(f"Loading Classification dataset: {dataset_name}, train_split: {train_split}, eval_split: {split}")
        try:
            if dataset_name == "sst2":
                dataset_train = load_dataset("glue", "sst2", split=train_split, cache_dir=self.cache_dir)
                dataset_eval = load_dataset("glue", "sst2", split=split, cache_dir=self.cache_dir)
                text_key = text_key or "sentence"
            elif dataset_name == "imdb":
                dataset_train = load_dataset("imdb", split=train_split, cache_dir=self.cache_dir)
                dataset_eval = load_dataset("imdb", split=split, cache_dir=self.cache_dir)
                text_key = text_key or "text"
            elif dataset_name == "ag_news":
                 dataset_train = load_dataset("ag_news", split=train_split, cache_dir=self.cache_dir)
                 dataset_eval = load_dataset("ag_news", split=split, cache_dir=self.cache_dir) # Often 'test'
                 text_key = text_key or "text"
            elif dataset_name == "trec":
                 dataset_train = load_dataset("trec", split=train_split, cache_dir=self.cache_dir)
                 dataset_eval = load_dataset("trec", split=split, cache_dir=self.cache_dir) # Often 'test'
                 text_key = text_key or "text"
                 label_key = "label-coarse" # TREC has fine and coarse labels
            # Add more classification datasets here (e.g., MRPC, QQP, MNLI would need pair handling)
            else:
                # Attempt direct load
                logger.warning(f"Attempting direct load for '{dataset_name}'. Specify text_key if needed.")
                dataset_train = load_dataset(dataset_name, split=train_split, cache_dir=self.cache_dir)
                dataset_eval = load_dataset(dataset_name, split=split, cache_dir=self.cache_dir)
                if text_key is None:
                    raise ValueError("text_key must be specified for non-standard classification datasets.")

        except Exception as e:
             logger.error(f"Failed to load classification dataset {dataset_name}: {e}")
             return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        # --- Data Sampling and Extraction ---
        try:
            # Limit training samples
            if max_train_samples and len(dataset_train) > max_train_samples:
                logger.info(f"Sampling {max_train_samples} for classifier training.")
                dataset_train = dataset_train.shuffle(seed=42).select(range(max_train_samples))

            # Limit evaluation samples
            if max_eval_samples and len(dataset_eval) > max_eval_samples:
                logger.info(f"Sampling {max_eval_samples} for classifier evaluation.")
                dataset_eval = dataset_eval.shuffle(seed=42).select(range(max_eval_samples))

            train_texts = dataset_train[text_key]
            train_labels = np.array(dataset_train[label_key])
            eval_texts = dataset_eval[text_key]
            eval_labels = np.array(dataset_eval[label_key])
        except KeyError as e:
             logger.error(f"Missing expected column in {dataset_name}: {e}. Columns: {dataset_train.column_names}")
             return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        except Exception as e:
             logger.error(f"Error during data extraction/sampling for {dataset_name}: {e}")
             return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        # --- Encoding ---
        logger.info(f"Encoding {len(train_texts)} training texts for {dataset_name}...")
        start_time = time.time()
        train_embeddings = self._encode_batch(train_texts, batch_size)
        encoding_time = time.time() - start_time
        logger.info(f"Training encoding finished in {encoding_time:.2f} seconds.")

        logger.info(f"Encoding {len(eval_texts)} evaluation texts for {dataset_name}...")
        start_time = time.time()
        eval_embeddings = self._encode_batch(eval_texts, batch_size)
        encoding_time = time.time() - start_time
        logger.info(f"Evaluation encoding finished in {encoding_time:.2f} seconds.")

        if train_embeddings.shape[0] == 0 or eval_embeddings.shape[0] == 0:
             logger.error(f"Encoding produced empty results for {dataset_name}.")
             return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        # --- Train Logistic Regression Classifier ---
        logger.info("Training logistic regression classifier...")
        start_time = time.time()
        # Increase max_iter for potentially better convergence
        classifier = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        try:
            classifier.fit(train_embeddings, train_labels)
        except Exception as e:
             logger.error(f"Failed to train logistic regression classifier: {e}")
             return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        training_time = time.time() - start_time
        logger.info(f"Classifier training finished in {training_time:.2f} seconds.")

        # --- Evaluate Classifier ---
        logger.info("Evaluating classifier...")
        try:
            predictions = classifier.predict(eval_embeddings)
        except Exception as e:
             logger.error(f"Failed to predict with logistic regression classifier: {e}")
             return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}


        # --- Calculate Metrics ---
        try:
            accuracy = accuracy_score(eval_labels, predictions)
            # Use weighted average for multi-class, binary otherwise is fine
            avg_mode = 'weighted' if len(np.unique(train_labels)) > 2 else 'binary'
            precision, recall, f1, _ = precision_recall_fscore_support(eval_labels, predictions, average=avg_mode, zero_division=0)
        except Exception as e:
             logger.error(f"Failed to calculate classification metrics: {e}")
             return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        results = {
            "accuracy": accuracy * 100,  # Convert to percentage
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100
        }

        logger.info(f"Results for {dataset_name}: Accuracy={results['accuracy']:.2f}%, F1={results['f1']:.2f}%")

        if return_results:
            return results, {
                "eval_texts": eval_texts,
                "eval_labels": eval_labels.tolist(),
                "predictions": predictions.tolist(),
                "train_embeddings": train_embeddings,
                "eval_embeddings": eval_embeddings,
            }

        return results

    def evaluate_classification_all(self, batch_size=32, max_train_samples=10000, max_eval_samples=None, datasets=None):
        """
        Evaluate on multiple classification benchmarks.

        Args:
            batch_size: Batch size for encoding.
            max_train_samples: Maximum number of training samples for classifier.
            max_eval_samples: Maximum number of evaluation samples.
            datasets: List of classification dataset names. Defaults to a standard set.

        Returns:
            Dictionary with metrics for all specified datasets and an average.
        """
        if datasets is None:
            # Common classification benchmarks
            datasets = ["sst2", "imdb", "ag_news", "trec"]

        all_results = {}
        valid_results_count = 0
        total_accuracy = 0.0
        total_f1 = 0.0

        for dataset_name in datasets:
            try:
                # Determine appropriate splits (may vary per dataset)
                train_split = 'train'
                eval_split = 'validation' if dataset_name in ['sst2'] else 'test' # Adjust as needed

                dataset_results = self.evaluate_classification(
                    dataset_name=dataset_name,
                    split=eval_split,
                    train_split=train_split,
                    batch_size=batch_size,
                    max_train_samples=max_train_samples,
                    max_eval_samples=max_eval_samples
                )

                if abs(dataset_results["accuracy"]) > 1e-6 or abs(dataset_results["f1"]) > 1e-6:
                     all_results[dataset_name] = dataset_results
                     total_accuracy += dataset_results["accuracy"]
                     total_f1 += dataset_results["f1"]
                     valid_results_count += 1
                else:
                    logger.warning(f"Skipping {dataset_name} results in average due to likely failure (scores are zero).")
                    all_results[dataset_name] = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

            except Exception as e:
                logger.error(f"Error evaluating classification dataset {dataset_name}: {str(e)}")
                all_results[dataset_name] = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        # Calculate average over datasets with valid results
        avg_accuracy = (total_accuracy / valid_results_count) if valid_results_count > 0 else 0.0
        avg_f1 = (total_f1 / valid_results_count) if valid_results_count > 0 else 0.0

        all_results["average"] = {
            "accuracy": avg_accuracy,
            "f1": avg_f1
        }
        logger.info(f"Classification Average (over {valid_results_count} datasets): Accuracy={avg_accuracy:.2f}%, F1={avg_f1:.2f}%")

        return all_results

    # --- Placeholder for Retrieval Evaluation ---
    # Retrieval evaluation (e.g., MSMARCO, NQ) is more complex, involving:
    # 1. Loading queries, corpus (documents/passages), and relevance judgments (qrels).
    # 2. Encoding all queries and the entire corpus (can be very large).
    # 3. Performing efficient similarity search (e.g., using FAISS or Annoy) for each query against the corpus.
    # 4. Calculating metrics like MRR@k, Recall@k, NDCG@k based on the rankings and qrels.
    # Implementing this robustly is beyond a quick script modification.
    # Consider using established benchmark frameworks like MTEB (Massive Text Embedding Benchmark)
    # or BEIR (Benchmarking-Effective-Information-Retrieval) for standardized retrieval evaluation.

    def evaluate_retrieval(self, *args, **kwargs):
         logger.warning("Retrieval evaluation is complex and not fully implemented here. "
                       "Consider using MTEB or BEIR frameworks for standardized evaluation.")
         # Return dummy results
         return {"average": {"mrr": 0.0, "recall": 0.0, "ndcg": 0.0}}


    def evaluate_clustering(self,
                           dataset_name="TwentyNewsgroupsClustering", # Example from MTEB
                           split="test",
                           batch_size=32,
                           max_samples=2000, # Clustering can be slow on large N
                           return_results=False):
        """
        Evaluate embedding quality for clustering using K-Means.

        Args:
            dataset_name: Name of the clustering dataset (e.g., from MTEB list).
            split: Data split to evaluate on.
            batch_size: Batch size for encoding.
            max_samples: Maximum number of samples to use for clustering.
            return_results: Whether to return detailed results.

        Returns:
            Dictionary with clustering metrics (ARI, NMI), optionally detailed results.
        """

        # --- Dataset Loading ---
        logger.info(f"Loading Clustering dataset: {dataset_name}, split: {split}")
        # Use MTEB names for consistency if possible
        try:
             # Example: Load a dataset structured like MTEB clustering tasks
             # Assumes dataset has 'sentences' and 'labels' columns
             dataset = load_dataset(f"mteb/{dataset_name}", split=split, cache_dir=self.cache_dir)
             text_key = 'sentences' # MTEB standard? Check dataset specifics
             label_key = 'labels'

        except Exception as e:
             # Fallback for other common clustering datasets
             if dataset_name == "ag_news_clustering": # Fictional example name
                 dataset = load_dataset("ag_news", split="test", cache_dir=self.cache_dir)
                 text_key = "text"
                 label_key = "label"
             elif dataset_name == "wikitext103_clustering": # Fictional example
                 dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", cache_dir=self.cache_dir)
                 # WikiText doesn't have inherent labels for clustering eval, might need custom setup
                 logger.error("Wikitext103 does not have standard labels for clustering evaluation.")
                 return {"ari": 0.0, "nmi": 0.0}
             else:
                 logger.error(f"Failed to load or identify structure for clustering dataset {dataset_name}: {e}")
                 return {"ari": 0.0, "nmi": 0.0}

        # --- Data Sampling and Extraction ---
        try:
            # Limit samples for performance
            if max_samples and len(dataset) > max_samples:
                logger.info(f"Sampling {max_samples} for clustering evaluation.")
                dataset = dataset.shuffle(seed=42).select(range(max_samples))

            texts = dataset[text_key]
            # Ensure texts are lists of strings if dataset structure varies
            if isinstance(texts[0], list): # Handle datasets where 'sentences' might be lists
                 texts = [item for sublist in texts for item in sublist] # Flatten
                 # Need to adjust labels accordingly - this gets complex if labels were per-list
                 logger.warning(f"Dataset '{dataset_name}' appears to have nested sentences. Clustering eval might be inaccurate if labels don't match flattened structure.")
                 # For simplicity, assuming labels correspond to flattened structure or use 1 label per original entry
                 labels = np.array(dataset[label_key])
                 if len(labels) != len(texts):
                     # Attempt to repeat labels if appropriate, otherwise fail
                     if len(dataset[label_key]) == len(dataset): # If labels were per outer list
                          labels = np.repeat(dataset[label_key], [len(s) for s in dataset[text_key]])
                          if len(labels) != len(texts):
                              logger.error("Label mismatch after flattening sentences.")
                              return {"ari": 0.0, "nmi": 0.0}
                     else:
                          logger.error("Label structure mismatch with sentence structure.")
                          return {"ari": 0.0, "nmi": 0.0}

            else:
                 labels = np.array(dataset[label_key])

            # Ensure texts are strings
            texts = [str(t) for t in texts]

            if len(texts) == 0:
                 logger.error("No texts found after sampling/extraction.")
                 return {"ari": 0.0, "nmi": 0.0}

        except KeyError as e:
             logger.error(f"Missing expected column in {dataset_name}: {e}. Columns: {dataset.column_names}")
             return {"ari": 0.0, "nmi": 0.0}
        except Exception as e:
             logger.error(f"Error during data extraction/sampling for clustering {dataset_name}: {e}")
             return {"ari": 0.0, "nmi": 0.0}


        # --- Encoding ---
        logger.info(f"Encoding {len(texts)} texts for clustering ({dataset_name})...")
        start_time = time.time()
        embeddings = self._encode_batch(texts, batch_size)
        encoding_time = time.time() - start_time
        logger.info(f"Clustering encoding finished in {encoding_time:.2f} seconds.")

        if embeddings.shape[0] == 0:
             logger.error(f"Encoding produced empty results for clustering {dataset_name}.")
             return {"ari": 0.0, "nmi": 0.0}

        # --- Perform K-means Clustering ---
        n_clusters = len(np.unique(labels))
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        start_time = time.time()
        # n_init='auto' is preferred in newer sklearn versions
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        try:
            cluster_labels = kmeans.fit_predict(embeddings)
        except Exception as e:
             logger.error(f"K-means clustering failed: {e}")
             return {"ari": 0.0, "nmi": 0.0}
        clustering_time = time.time() - start_time
        logger.info(f"K-means finished in {clustering_time:.2f} seconds.")

        # --- Calculate Metrics ---
        try:
            ari = adjusted_rand_score(labels, cluster_labels)
            nmi = normalized_mutual_info_score(labels, cluster_labels)
        except Exception as e:
             logger.error(f"Failed to calculate clustering metrics: {e}")
             return {"ari": 0.0, "nmi": 0.0}


        results = {
            "ari": ari * 100,  # Convert to percentage
            "nmi": nmi * 100
        }

        logger.info(f"Results for {dataset_name}: ARI={results['ari']:.2f}%, NMI={results['nmi']:.2f}%")

        if return_results:
            return results, {
                "texts": texts,
                "true_labels": labels.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "embeddings": embeddings,
            }

        return results

    def evaluate_clustering_all(self, batch_size=32, max_samples=2000, datasets=None):
        """ Evaluate on multiple clustering benchmarks. """
        if datasets is None:
             # Example clustering datasets from MTEB
             datasets = [
                  "TwentyNewsgroupsClustering",
                  "ArxivClusteringP2P", # P2P = Title+Abstract vs Title+Abstract
                  "BiorxivClusteringP2P",
                  # Add more relevant datasets
             ]

        all_results = {}
        valid_results_count = 0
        total_ari = 0.0
        total_nmi = 0.0

        for dataset_name in datasets:
            try:
                dataset_results = self.evaluate_clustering(
                    dataset_name=dataset_name,
                    split="test", # Usually 'test' for clustering benchmarks
                    batch_size=batch_size,
                    max_samples=max_samples
                )
                if abs(dataset_results["ari"]) > 1e-6 or abs(dataset_results["nmi"]) > 1e-6:
                    all_results[dataset_name] = dataset_results
                    total_ari += dataset_results["ari"]
                    total_nmi += dataset_results["nmi"]
                    valid_results_count += 1
                else:
                    logger.warning(f"Skipping clustering {dataset_name} results in average due to likely failure.")
                    all_results[dataset_name] = {"ari": 0.0, "nmi": 0.0}
            except Exception as e:
                logger.error(f"Error evaluating clustering dataset {dataset_name}: {str(e)}")
                all_results[dataset_name] = {"ari": 0.0, "nmi": 0.0}

        avg_ari = (total_ari / valid_results_count) if valid_results_count > 0 else 0.0
        avg_nmi = (total_nmi / valid_results_count) if valid_results_count > 0 else 0.0

        all_results["average"] = {
            "ari": avg_ari,
            "nmi": avg_nmi
        }
        logger.info(f"Clustering Average (over {valid_results_count} datasets): ARI={avg_ari:.2f}%, NMI={avg_nmi:.2f}%")
        return all_results


    def evaluate_all(self, batch_size=32, max_samples_cluster=2000, max_train_samples_classif=10000, max_eval_samples_classif=None):
        """
        Run evaluations for STS, Classification, and Clustering.

        Returns:
            Dictionary containing results for all evaluation categories.
        """
        all_results = {}
        start_time_total = time.time()

        # STS evaluation
        logger.info("--- Starting STS Evaluation ---")
        start_time = time.time()
        all_results["sts"] = self.evaluate_sts_all(batch_size=batch_size)
        logger.info(f"STS evaluation took {time.time() - start_time:.2f} seconds.")

        # Classification evaluation
        logger.info("--- Starting Classification Evaluation ---")
        start_time = time.time()
        all_results["classification"] = self.evaluate_classification_all(
            batch_size=batch_size,
            max_train_samples=max_train_samples_classif,
            max_eval_samples=max_eval_samples_classif
        )
        logger.info(f"Classification evaluation took {time.time() - start_time:.2f} seconds.")

        # Clustering evaluation
        logger.info("--- Starting Clustering Evaluation ---")
        start_time = time.time()
        try:
            all_results["clustering"] = self.evaluate_clustering_all(
                 batch_size=batch_size,
                 max_samples=max_samples_cluster
            )
        except Exception as e:
            logger.error(f"Clustering evaluation failed: {str(e)}")
            all_results["clustering"] = {"average": {"ari": 0.0, "nmi": 0.0}} # Default failure
        logger.info(f"Clustering evaluation took {time.time() - start_time:.2f} seconds.")

        # Retrieval evaluation (Placeholder)
        logger.info("--- Starting Retrieval Evaluation (Placeholder) ---")
        start_time = time.time()
        all_results["retrieval"] = self.evaluate_retrieval()
        logger.info(f"Retrieval evaluation took {time.time() - start_time:.2f} seconds.")


        # Calculate overall average (simple mean of primary metrics)
        sts_avg = all_results.get("sts", {}).get("average", {}).get("spearman", 0.0)
        classif_avg = all_results.get("classification", {}).get("average", {}).get("accuracy", 0.0)
        cluster_avg = all_results.get("clustering", {}).get("average", {}).get("nmi", 0.0)
        retrieval_avg = all_results.get("retrieval", {}).get("average", {}).get("mrr", 0.0) # Using MRR as example

        # Count how many task types had valid results
        valid_task_averages = [avg for avg in [sts_avg, classif_avg, cluster_avg, retrieval_avg] if abs(avg) > 1e-6]
        overall_average = sum(valid_task_averages) / len(valid_task_averages) if valid_task_averages else 0.0

        all_results["overall_average"] = overall_average
        logger.info(f"--- Overall Evaluation Summary ---")
        logger.info(f"Average STS (Spearman): {sts_avg:.2f}%")
        logger.info(f"Average Classification (Accuracy): {classif_avg:.2f}%")
        logger.info(f"Average Clustering (NMI): {cluster_avg:.2f}%")
        logger.info(f"Average Retrieval (MRR - Placeholder): {retrieval_avg:.2f}%")
        logger.info(f"Overall Average Score: {overall_average:.2f}%")
        logger.info(f"Total evaluation time: {time.time() - start_time_total:.2f} seconds.")

        return all_results

    def visualize_embeddings(self,
                            dataset_name="sst2", # Dataset to get texts and labels from
                            split="validation",
                            max_samples=1000,
                            batch_size=32,
                            output_path="embedding_visualization.png",
                            use_tsne=True, # Option to use UMAP later
                            tsne_perplexity=30):
        """
        Visualize the embeddings using t-SNE (or potentially UMAP).

        Args:
            dataset_name: Name of the dataset for visualization (needs text and labels).
            split: Data split to use.
            max_samples: Maximum number of samples to visualize.
            batch_size: Batch size for encoding.
            output_path: Path to save the visualization PNG file.
            use_tsne: If True, use t-SNE; otherwise might use UMAP (not implemented yet).
            tsne_perplexity: Perplexity value for t-SNE.

        Returns:
            Path to the saved visualization file.
        """
        from sklearn.manifold import TSNE
        # from umap import UMAP # If using UMAP

        # --- Load Dataset ---
        logger.info(f"Loading dataset {dataset_name} for visualization...")
        # Simplified loading - assumes binary classification for color mapping
        label_names = ["Class 0", "Class 1"] # Default names
        try:
            if dataset_name == "sst2":
                dataset = load_dataset("glue", "sst2", split=split, cache_dir=self.cache_dir)
                text_key, label_key = "sentence", "label"
                label_names = ["Negative", "Positive"]
            elif dataset_name == "imdb":
                dataset = load_dataset("imdb", split=split, cache_dir=self.cache_dir)
                text_key, label_key = "text", "label"
                label_names = ["Negative", "Positive"]
            elif dataset_name == "ag_news":
                 dataset = load_dataset("ag_news", split='test', cache_dir=self.cache_dir) # AG News has 4 classes
                 text_key, label_key = "text", "label"
                 label_names = ["World", "Sports", "Business", "Sci/Tech"]
            # Add more datasets suitable for visualization
            else:
                logger.warning(f"Visualization for '{dataset_name}' might not be optimal. Using default keys/labels.")
                dataset = load_dataset(dataset_name, split=split, cache_dir=self.cache_dir)
                text_key = "text" # Guessing
                label_key = "label" # Guessing
                num_labels = len(dataset.unique(label_key))
                label_names = [f"Class {i}" for i in range(num_labels)]


        except Exception as e:
             logger.error(f"Failed to load dataset {dataset_name} for visualization: {e}")
             return None

        # --- Sample and Extract Data ---
        try:
            if max_samples and len(dataset) > max_samples:
                logger.info(f"Sampling {max_samples} for visualization.")
                dataset = dataset.shuffle(seed=42).select(range(max_samples))

            texts = dataset[text_key]
            labels = np.array(dataset[label_key])
            texts = [str(t) for t in texts] # Ensure strings

            if len(texts) == 0:
                 logger.error("No texts found for visualization.")
                 return None

        except Exception as e:
             logger.error(f"Error extracting data for visualization: {e}")
             return None

        # --- Encode Texts ---
        logger.info(f"Encoding {len(texts)} texts for visualization...")
        embeddings = self._encode_batch(texts, batch_size)

        if embeddings.shape[0] == 0:
             logger.error("Encoding produced empty results for visualization.")
             return None

        # --- Dimensionality Reduction ---
        logger.info(f"Applying {'t-SNE' if use_tsne else 'UMAP'}...")
        start_time = time.time()
        if use_tsne:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(tsne_perplexity, len(embeddings)-1), n_iter=1000, init='pca', learning_rate='auto')
            embeddings_2d = tsne.fit_transform(embeddings)
        else:
            # umap_model = UMAP(n_components=2, random_state=42)
            # embeddings_2d = umap_model.fit_transform(embeddings)
             logger.warning("UMAP visualization not implemented yet, using t-SNE.")
             tsne = TSNE(n_components=2, random_state=42, perplexity=min(tsne_perplexity, len(embeddings)-1), n_iter=1000, init='pca', learning_rate='auto')
             embeddings_2d = tsne.fit_transform(embeddings)
        dr_time = time.time() - start_time
        logger.info(f"Dimensionality reduction took {dr_time:.2f} seconds.")

        # --- Plotting ---
        logger.info(f"Creating plot and saving to {output_path}")
        try:
            df = pd.DataFrame({
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                # Map numerical labels to names
                "label": [label_names[l] if 0 <= l < len(label_names) else f"Class {l}" for l in labels]
            })

            plt.figure(figsize=(12, 10))
            num_labels = len(df['label'].unique())
            palette = sns.color_palette("husl", n_colors=num_labels) # Use a suitable palette

            sns.scatterplot(data=df, x="x", y="y", hue="label", palette=palette, alpha=0.7, s=50) # Increased point size

            plt.title(f"{'t-SNE' if use_tsne else 'UMAP'} Visualization of '{self.model_name}' Embeddings on {dataset_name.upper()}", fontsize=16)
            plt.xlabel("Dimension 1", fontsize=12)
            plt.ylabel("Dimension 2", fontsize=12)
            plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close() # Close the plot to free memory

            logger.info(f"Visualization saved successfully to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to create or save visualization: {e}")
            plt.close() # Ensure plot is closed even on error
            return None


# --- Comparison Functions ---
def compare_models(models_dict: Dict[str, object], # Dict mapping name -> model instance
                   output_dir: str,
                   batch_size: int = 32,
                   max_samples_cluster: int = 2000,
                   max_train_samples_classif: int = 10000,
                   max_eval_samples_classif: int = None,
                   run_visualization: bool = True,
                   vis_dataset: str = "sst2",
                   vis_max_samples: int = 1000):
    """
    Compare multiple sentence embedding models on evaluation benchmarks.

    Args:
        models_dict: Dictionary mapping model names (str) to model instances.
                     Each model instance must have an `.encode()` method.
        output_dir: Directory to save results and plots.
        batch_size: Batch size for encoding during evaluation.
        max_samples_cluster: Max samples for clustering evaluation.
        max_train_samples_classif: Max samples for training classification probes.
        max_eval_samples_classif: Max samples for evaluating classification probes.
        run_visualization: Whether to generate t-SNE plots.
        vis_dataset: Dataset to use for t-SNE visualization.
        vis_max_samples: Max samples for t-SNE visualization.

    Returns:
        Dictionary containing comprehensive evaluation results for all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    comparison_start_time = time.time()

    for model_name, model_instance in models_dict.items():
        logger.info(f"===== Evaluating model: {model_name} =====")
        model_start_time = time.time()

        # Initialize evaluator for the current model
        evaluator = SentenceEmbeddingEvaluator(model=model_instance, model_name_or_path=model_name)

        # Run all standard evaluations
        try:
             model_results = evaluator.evaluate_all(
                 batch_size=batch_size,
                 max_samples_cluster=max_samples_cluster,
                 max_train_samples_classif=max_train_samples_classif,
                 max_eval_samples_classif=max_eval_samples_classif
             )
             all_results[model_name] = model_results

             # Save individual model results to JSON
             model_results_path = os.path.join(output_dir, f"{model_name}_results.json")
             try:
                 with open(model_results_path, "w") as f:
                     # Convert numpy arrays if any were returned (shouldn't be if return_results=False)
                     json.dump(model_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
                 logger.info(f"Saved individual results for {model_name} to {model_results_path}")
             except Exception as e:
                 logger.error(f"Failed to save results JSON for {model_name}: {e}")

        except Exception as e:
             logger.error(f"Evaluation failed entirely for model {model_name}: {e}")
             all_results[model_name] = {"error": str(e)} # Record the error


        # Run visualization if requested
        if run_visualization:
            logger.info(f"--- Generating Visualization for {model_name} ---")
            vis_path = os.path.join(output_dir, f"{model_name}_visualization.png")
            try:
                evaluator.visualize_embeddings(
                    dataset_name=vis_dataset,
                    split="validation", # Or appropriate split for vis_dataset
                    max_samples=vis_max_samples,
                    batch_size=batch_size,
                    output_path=vis_path
                )
                # Log visualization to wandb if available
                if _wandb_available and wandb.run and os.path.exists(vis_path):
                     wandb.log({f"visualizations/{model_name}": wandb.Image(vis_path)})

            except Exception as e:
                logger.error(f"Error creating visualization for {model_name}: {str(e)}")

        model_eval_time = time.time() - model_start_time
        logger.info(f"===== Finished evaluating {model_name} in {model_eval_time:.2f} seconds =====")


    # Save overall comparison results
    overall_results_path = os.path.join(output_dir, "comparison_results.json")
    try:
        with open(overall_results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        logger.info(f"Saved overall comparison results to {overall_results_path}")
    except Exception as e:
        logger.error(f"Failed to save overall comparison JSON: {e}")

    # Create comparison plots
    logger.info("--- Generating Comparison Plots ---")
    try:
         plot_paths = create_comparison_plots(all_results, output_dir)
         # Log plots to wandb
         if _wandb_available and wandb.run:
              for plot_name, plot_path in plot_paths.items():
                   if os.path.exists(plot_path):
                        wandb.log({f"comparison_plots/{plot_name}": wandb.Image(plot_path)})
    except Exception as e:
         logger.error(f"Failed to generate comparison plots: {e}")


    total_comparison_time = time.time() - comparison_start_time
    logger.info(f"===== Total comparison finished in {total_comparison_time:.2f} seconds =====")

    return all_results


def create_comparison_plots(results: Dict[str, Dict], output_dir: str) -> Dict[str, str]:
    """
    Create comparison bar plots and a radar plot for evaluated models.

    Args:
        results: Dictionary with evaluation results for all models.
                 Structure: {model_name: {task_type: {dataset/average: {metric: score}}}}
        output_dir: Directory to save plots.

    Returns:
        Dictionary mapping plot names to their file paths.
    """
    model_names = list(results.keys())
    plot_paths = {}

    if not model_names:
        logger.warning("No model results found to create comparison plots.")
        return plot_paths

    # Helper function to safely get scores
    def get_score(model_name, task, sub_task, metric, default=0.0):
        try:
            return results[model_name][task][sub_task][metric]
        except KeyError:
            # logger.debug(f"Metric {task}/{sub_task}/{metric} not found for {model_name}. Using default {default}.")
            return default
        except TypeError: # Handle case where results[model_name] might be {"error": ...}
            # logger.debug(f"Results structure invalid for {model_name}. Using default {default}.")
            return default


    # --- Plot STS results (Average Spearman) ---
    try:
        plt.figure(figsize=(max(6, len(model_names) * 0.8), 6))
        sts_spearman = [get_score(model, "sts", "average", "spearman") for model in model_names]
        bars = plt.bar(model_names, sts_spearman, color=sns.color_palette("viridis", len(model_names)))
        plt.ylabel("Average Spearman Correlation (%)", fontsize=12)
        plt.title("Semantic Textual Similarity Performance", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(bottom=max(0, min(sts_spearman)-10), top=max(80, max(sts_spearman)+5)) # Adjust ylim dynamically
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Add labels on bars
        for bar in bars:
             yval = bar.get_height()
             plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}', va='bottom', ha='center', fontsize=9) # va='bottom' to place above bar

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "sts_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plot_paths["sts_comparison"] = plot_path
        plt.close()
        logger.info(f"Saved STS comparison plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to create STS plot: {e}")
        plt.close()

    # --- Plot Classification results (Average Accuracy) ---
    try:
        plt.figure(figsize=(max(6, len(model_names) * 0.8), 6))
        classif_acc = [get_score(model, "classification", "average", "accuracy") for model in model_names]
        bars = plt.bar(model_names, classif_acc, color=sns.color_palette("magma", len(model_names)))
        plt.ylabel("Average Accuracy (%)", fontsize=12)
        plt.title("Classification Performance", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(bottom=max(0, min(classif_acc)-10), top=max(90, max(classif_acc)+5))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
             yval = bar.get_height()
             plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}', va='bottom', ha='center', fontsize=9)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "classification_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plot_paths["classification_comparison"] = plot_path
        plt.close()
        logger.info(f"Saved Classification comparison plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to create Classification plot: {e}")
        plt.close()


    # --- Plot Clustering results (Average NMI) ---
    try:
        plt.figure(figsize=(max(6, len(model_names) * 0.8), 6))
        cluster_nmi = [get_score(model, "clustering", "average", "nmi") for model in model_names]
        bars = plt.bar(model_names, cluster_nmi, color=sns.color_palette("plasma", len(model_names)))
        plt.ylabel("Average Normalized Mutual Information (NMI %)", fontsize=12)
        plt.title("Clustering Performance", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim(bottom=max(0, min(cluster_nmi)-10), top=max(70, max(cluster_nmi)+5))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
             yval = bar.get_height()
             plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}', va='bottom', ha='center', fontsize=9)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "clustering_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plot_paths["clustering_comparison"] = plot_path
        plt.close()
        logger.info(f"Saved Clustering comparison plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to create Clustering plot: {e}")
        plt.close()


    # --- Overall performance radar plot ---
    try:
        metrics_radar = ["STS", "Classification", "Clustering", "Retrieval"]
        # Get average scores, normalize to 0-1 (assuming scores are percentages)
        # Handle potential KeyError gracefully
        values_radar = {}
        for model in model_names:
             sts_val = get_score(model, "sts", "average", "spearman", 0) / 100.0
             cls_val = get_score(model, "classification", "average", "accuracy", 0) / 100.0
             clu_val = get_score(model, "clustering", "average", "nmi", 0) / 100.0
             ret_val = get_score(model, "retrieval", "average", "mrr", 0) / 100.0 # Using MRR
             values_radar[model] = [sts_val, cls_val, clu_val, ret_val]


        N = len(metrics_radar)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1] # Close the loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Set y-axis limits (0 to 1 after normalization)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 5)) # Add ticks for 0, 0.25, 0.5, 0.75, 1.0
        ax.set_yticklabels([f"{i*25:.0f}%" for i in range(5)]) # Label ticks as percentages


        # Plot data for each model
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for i, (model, model_values) in enumerate(values_radar.items()):
            data = model_values + model_values[:1] # Close the loop
            color = colors[i % len(colors)]
            ax.plot(angles, data, linewidth=2, linestyle='solid', label=model, color=color, marker='o')
            ax.fill(angles, data, color=color, alpha=0.2)

        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_radar, fontsize=12)


        plt.title("Overall Model Performance Comparison", size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)) # Adjust legend position

        plot_path = os.path.join(output_dir, "overall_comparison_radar.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_paths["overall_comparison_radar"] = plot_path
        plt.close()
        logger.info(f"Saved Overall comparison radar plot to {plot_path}")

    except Exception as e:
        logger.error(f"Failed to create Radar plot: {e}")
        plt.close()


    return plot_paths