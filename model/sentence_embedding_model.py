# model/sentence_embedding_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import os
import json

# Try importing PEFT, fail gracefully if not installed
try:
    from peft import get_peft_model, LoraConfig, TaskType
    _peft_available = True
except ImportError:
    _peft_available = False
    LoraConfig = None # Define as None if PEFT not available
    TaskType = None
    get_peft_model = None


class SentenceEmbeddingAutoencoder(nn.Module):
    """
    Sentence embedding autoencoder model that uses a transformer encoder to produce
    a fixed-size sentence embedding, then uses a decoder with blank tokens to reconstruct
    the original input.
    """
    def __init__(
        self,
        encoder_model_name: str = "meta-llama/Llama-2-7b",
        decoder_model_name: str = "meta-llama/Llama-2-7b",
        max_length: int = 128,
        embedding_dim: int = 4096,  # Default for Llama 2
        pooling_strategy: str = "last_token",  # or "mean", "cls", etc.
        noise_std: float = 0.1,
        noise_strategy: str = "gaussian"  # or "dropout", "zeros"
    ):
        super().__init__()

        # Initialize encoder and decoder with pre-trained models
        self.encoder = AutoModelForCausalLM.from_pretrained(encoder_model_name)
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name)

        # Tokenizers
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

        if self.encoder_tokenizer.pad_token is None:
            self.encoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.encoder.resize_token_embeddings(len(self.encoder_tokenizer))
            print("Added PAD token to encoder tokenizer and resized embeddings.")
        if self.decoder_tokenizer.pad_token is None:
             self.decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
             print("Added PAD token to decoder tokenizer and resized embeddings.")


        # Configuration - store for saving/loading
        self.config = {
            "encoder_model_name": encoder_model_name,
            "decoder_model_name": decoder_model_name,
            "max_length": max_length,
            "embedding_dim": embedding_dim,
            "pooling_strategy": pooling_strategy,
            "noise_std": noise_std,
            "noise_strategy": noise_strategy,
        }

        self.max_length = max_length
        self.embedding_dim = embedding_dim # This might be overridden by actual model dim
        self.pooling_strategy = pooling_strategy
        self.noise_std = noise_std
        self.noise_strategy = noise_strategy

        # Projection layer to ensure encoder-decoder dimensionality match
        encoder_hidden_size = self.encoder.config.hidden_size
        decoder_hidden_size = self.decoder.config.hidden_size
        self.embedding_dim = decoder_hidden_size # Use actual decoder hidden size as embedding dim

        if encoder_hidden_size != decoder_hidden_size:
            self.projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.projection = nn.Identity()

    def get_sentence_embedding(self, input_ids, attention_mask=None):
        """
        Extract a sentence embedding from the encoder model using the specified pooling strategy.
        """
        # Get the outputs from the encoder
        # Note: Llama is causal, so output_hidden_states=True gives all layer states
        # We typically use the final layer's hidden states
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the last hidden states
        hidden_states = outputs.hidden_states[-1] # Shape: [batch_size, seq_len, hidden_size]

        if attention_mask is None:
             attention_mask = torch.ones_like(input_ids)

        # Apply pooling strategy
        if self.pooling_strategy == "mean":
            # Mean pooling over all non-padding tokens
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            sentence_embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9) # Avoid division by zero

        elif self.pooling_strategy == "cls":
            # Use the [CLS] token embedding (first token for BERT-like models)
            # For causal models, this is usually the first token but might not be trained as CLS
            print("Warning: Using 'cls' pooling strategy with a potentially causal model. Ensure the first token is meaningful.")
            sentence_embedding = hidden_states[:, 0]

        elif self.pooling_strategy == "last_token":
            # For autoregressive models like GPT/Llama, use the last non-padding token
            batch_size = hidden_states.shape[0]
            seq_lengths = attention_mask.sum(dim=1) - 1 # Indices are 0-based
            # Gather the hidden states at the last token position for each sequence
            sentence_embedding = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]

        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")

        # Apply projection if needed
        sentence_embedding = self.projection(sentence_embedding) # Shape: [batch_size, decoder_hidden_size]

        return sentence_embedding

    def generate_blank_tokens(self, sentence_embedding, seq_length):
        """
        Generate n-1 blank/noisy tokens, where n is the original sequence length.
        These will be concatenated with the sentence embedding for the decoder.
        """
        batch_size = sentence_embedding.shape[0]
        hidden_dim = sentence_embedding.shape[1] # Should be decoder_hidden_size

        # Create blank tokens with the specified noise strategy
        if self.noise_strategy == "gaussian":
            # Gaussian noise
            blank_tokens = torch.randn(batch_size, seq_length - 1, hidden_dim, device=sentence_embedding.device) * self.noise_std

        elif self.noise_strategy == "dropout":
            # Apply dropout to zero out some dimensions randomly - applied to noise
            # Create noise first, then apply dropout
            noise = torch.randn(batch_size, seq_length - 1, hidden_dim, device=sentence_embedding.device) * self.noise_std
            dropout = nn.Dropout(p=0.5) # Consider making dropout p configurable
            blank_tokens = dropout(noise)

        elif self.noise_strategy == "zeros":
            # Just zeros (completely blank)
            blank_tokens = torch.zeros(batch_size, seq_length - 1, hidden_dim, device=sentence_embedding.device)

        else:
            raise ValueError(f"Unsupported noise strategy: {self.noise_strategy}")

        return blank_tokens # Shape: [batch_size, seq_length-1, hidden_dim]

    def decode_from_embedding(self, sentence_embedding, target_ids, target_mask=None):
        """
        Decode the original sentence from the sentence embedding + blank tokens.
        Uses teacher forcing by providing inputs_embeds.
        """
        batch_size = sentence_embedding.shape[0]
        seq_length = target_ids.shape[1] # Use target sequence length

        # Generate blank/noisy tokens
        # Make sure seq_length > 0
        if seq_length <= 1:
            # If sequence length is 1 or less, we can't generate n-1 blank tokens.
            # Handle this case: perhaps just use the sentence embedding directly?
            # Or return an error/warning. For now, let's assume seq_length > 1.
            # If only 1 token, maybe decoder input is just sentence_embedding?
            # This needs careful thought based on task requirements.
            # Let's proceed assuming seq_length > 1 for the standard autoencoder case.
             if seq_length == 1:
                 decoder_inputs = sentence_embedding.unsqueeze(1)
             else:
                 # This case should ideally not happen with padding="max_length"
                 raise ValueError(f"Sequence length ({seq_length}) must be greater than 1 for this decoding strategy.")
        else:
            blank_tokens = self.generate_blank_tokens(sentence_embedding, seq_length)
            # Expand the sentence embedding to shape [batch_size, 1, hidden_dim]
            sentence_embedding_expanded = sentence_embedding.unsqueeze(1)
            # Concatenate the sentence embedding with the blank tokens
            # Shape: [batch_size, seq_length, hidden_dim]
            decoder_inputs = torch.cat([sentence_embedding_expanded, blank_tokens], dim=1)


        # Prepare labels for causal LM loss calculation
        # Shift labels: The prediction for token i should be based on inputs up to i-1
        # For causal LM, the loss is typically calculated on shifted labels.
        # `transformers` handles this internally if `labels` are provided.
        labels = target_ids.clone()
        if self.decoder_tokenizer.pad_token_id is not None:
            labels[labels == self.decoder_tokenizer.pad_token_id] = -100 # Ignore padding index in loss


        # Pass through the decoder using inputs_embeds for teacher forcing
        outputs = self.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=target_mask, # Use target mask for decoder attention
            labels=labels, # Let the model compute the loss
            return_dict=True
        )

        # Logits shape: [batch_size, seq_len, vocab_size]
        # Loss is calculated by the model based on logits and labels
        return outputs.logits, outputs.loss

    def forward(self, input_ids, attention_mask=None, target_ids=None, target_mask=None):
        """
        Forward pass through the full autoencoder model.

        For training, target_ids should typically be the same as input_ids.
        """
        # If target_ids not provided, use input_ids (standard autoencoder)
        if target_ids is None:
            target_ids = input_ids
            target_mask = attention_mask

        # Ensure masks exist if IDs are provided
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids)
        if target_mask is None and target_ids is not None:
             target_mask = torch.ones_like(target_ids)


        # Get sentence embedding from the encoder
        sentence_embedding = self.get_sentence_embedding(input_ids, attention_mask)

        # Decode to reconstruct the original sentence
        logits, loss = self.decode_from_embedding(sentence_embedding, target_ids, target_mask)

        return {
            "sentence_embedding": sentence_embedding,
            "logits": logits,
            "loss": loss # Loss is already calculated by the decoder model
        }

    def encode(self, sentences: Union[str, List[str]], batch_size=32, device="cuda"):
        """Encode a list of sentences into embeddings."""
        self.eval() # Ensure model is in eval mode
        single_sentence = False
        if isinstance(sentences, str):
            sentences = [sentences]
            single_sentence = True

        all_embeddings = []
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]

            # Tokenize
            inputs = self.encoder_tokenizer(
                batch,
                return_tensors="pt",
                padding=True, # Pad to longest in batch
                truncation=True,
                max_length=self.max_length
            ).to(device)

            # Encode
            with torch.no_grad():
                # Ensure the model is on the correct device
                self.to(device)
                embedding = self.get_sentence_embedding(
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )

            all_embeddings.append(embedding.cpu()) # Move to CPU to save GPU memory

        # Concatenate all batches
        final_embeddings = torch.cat(all_embeddings, dim=0)

        if single_sentence:
             return final_embeddings[0]
        return final_embeddings

    def save_pretrained(self, save_directory):
        """ Save model and tokenizer to a directory """
        os.makedirs(save_directory, exist_ok=True)
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f)
        # Save tokenizers
        self.encoder_tokenizer.save_pretrained(os.path.join(save_directory, "encoder_tokenizer"))
        self.decoder_tokenizer.save_pretrained(os.path.join(save_directory, "decoder_tokenizer"))

    @classmethod
    def from_pretrained(cls, load_directory):
        """ Load model and tokenizer from a directory """
        # Load config
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)

        # Initialize model with config
        model = cls(**config)

        # Load state dict
        model.load_state_dict(torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu"))

        # Load tokenizers (already done in __init__, but maybe re-load if needed? Check this)
        # model.encoder_tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "encoder_tokenizer"))
        # model.decoder_tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "decoder_tokenizer"))

        return model


class SentenceEmbeddingAutoencoderLoRA(SentenceEmbeddingAutoencoder):
    """
    Version of the SentenceEmbeddingAutoencoder that uses LoRA (Low-Rank Adaptation)
    for more efficient fine-tuning of the large language models.
    """
    def __init__(
        self,
        encoder_model_name: str = "meta-llama/Llama-2-7b",
        decoder_model_name: str = "meta-llama/Llama-2-7b",
        max_length: int = 128,
        embedding_dim: int = 4096,
        pooling_strategy: str = "last_token",
        noise_std: float = 0.1,
        noise_strategy: str = "gaussian",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None, # Allow specifying target modules
    ):
        if not _peft_available:
            raise ImportError("PEFT is not available. Please install it to use LoRA: pip install peft")

        # Initialize the base class structure *without* loading full models yet
        # We need nn.Module init first
        super(SentenceEmbeddingAutoencoder, self).__init__() # Call grand-parent's init directly

        # Store LoRA config parameters
        self.lora_config_params = {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules or ["q_proj", "v_proj"], # Default if None
        }

        # --- Re-do parts of the base init ---
        # Tokenizers
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

        # Configuration - store for saving/loading
        self.config = {
            "encoder_model_name": encoder_model_name,
            "decoder_model_name": decoder_model_name,
            "max_length": max_length,
            "embedding_dim": embedding_dim, # Will be updated
            "pooling_strategy": pooling_strategy,
            "noise_std": noise_std,
            "noise_strategy": noise_strategy,
            **self.lora_config_params # Add LoRA params to config
        }

        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        self.noise_std = noise_std
        self.noise_strategy = noise_strategy

        # --- Load base models ---
        # Load base models - consider memory usage, maybe load on meta device first if needed
        base_encoder = AutoModelForCausalLM.from_pretrained(encoder_model_name)
        base_decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name)

        # Handle pad tokens AFTER loading base models, BEFORE applying PEFT
        if self.encoder_tokenizer.pad_token is None:
            self.encoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            base_encoder.resize_token_embeddings(len(self.encoder_tokenizer))
            print("Added PAD token to encoder tokenizer and resized embeddings.")
        if self.decoder_tokenizer.pad_token is None:
             self.decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             base_decoder.resize_token_embeddings(len(self.decoder_tokenizer))
             print("Added PAD token to decoder tokenizer and resized embeddings.")


        # --- Apply LoRA ---
        # Create LoRA configs
        lora_config_shared = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=self.lora_config_params["lora_target_modules"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA adapters
        # Important: get_peft_model modifies the model in place if adapter_name exists
        self.encoder = get_peft_model(base_encoder, lora_config_shared, adapter_name="encoder_adapter")
        self.decoder = get_peft_model(base_decoder, lora_config_shared, adapter_name="decoder_adapter")

        print("Applied LoRA adapters to encoder and decoder.")
        self.encoder.print_trainable_parameters()
        self.decoder.print_trainable_parameters()

        # --- Finalize setup ---
        # Projection layer
        # Use config from the *base* model before PEFT wrapping
        encoder_hidden_size = self.encoder.base_model.config.hidden_size
        decoder_hidden_size = self.decoder.base_model.config.hidden_size
        self.embedding_dim = decoder_hidden_size # Update embedding dim

        if encoder_hidden_size != decoder_hidden_size:
            self.projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.projection = nn.Identity()

    # Override save/load to handle PEFT models
    def save_pretrained(self, save_directory):
        """ Save PEFT model adapters and config """
        os.makedirs(save_directory, exist_ok=True)

        # Save PEFT adapters
        self.encoder.save_pretrained(os.path.join(save_directory, "encoder_adapter"))
        self.decoder.save_pretrained(os.path.join(save_directory, "decoder_adapter"))

        # Save the projection layer state_dict separately if it exists and is not Identity
        if hasattr(self, 'projection') and not isinstance(self.projection, nn.Identity):
            torch.save(self.projection.state_dict(), os.path.join(save_directory, "projection.bin"))

        # Save overall config (including LoRA params)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f)

        # Save tokenizers
        self.encoder_tokenizer.save_pretrained(os.path.join(save_directory, "encoder_tokenizer"))
        self.decoder_tokenizer.save_pretrained(os.path.join(save_directory, "decoder_tokenizer"))

    @classmethod
    def from_pretrained(cls, load_directory):
        """ Load PEFT model adapters and config """
        if not _peft_available:
            raise ImportError("PEFT is not available. Please install it to load LoRA models: pip install peft")

        # Load overall config
        import json
        import os
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)

        # Extract base model names and LoRA params
        encoder_model_name = config["encoder_model_name"]
        decoder_model_name = config["decoder_model_name"]
        lora_params = {k: v for k, v in config.items() if k.startswith("lora_")}

        # Initialize the LoRA model class - this will load base models and apply NEW adapters
        # We will then load the saved adapter weights
        model = cls(
            encoder_model_name=encoder_model_name,
            decoder_model_name=decoder_model_name,
            max_length=config["max_length"],
            embedding_dim=config["embedding_dim"],
            pooling_strategy=config["pooling_strategy"],
            noise_std=config["noise_std"],
            noise_strategy=config["noise_strategy"],
            **lora_params
        )

        # --- Load the saved adapter weights ---
        # PEFT models load adapters using load_adapter
        encoder_adapter_path = os.path.join(load_directory, "encoder_adapter")
        decoder_adapter_path = os.path.join(load_directory, "decoder_adapter")

        if os.path.exists(encoder_adapter_path):
             print(f"Loading encoder adapter from {encoder_adapter_path}")
             # The adapter name should match the one used during init or saving
             model.encoder.load_adapter(encoder_adapter_path, adapter_name="encoder_adapter")
        else:
             print(f"Warning: Encoder adapter directory not found at {encoder_adapter_path}")

        if os.path.exists(decoder_adapter_path):
            print(f"Loading decoder adapter from {decoder_adapter_path}")
            model.decoder.load_adapter(decoder_adapter_path, adapter_name="decoder_adapter")
        else:
            print(f"Warning: Decoder adapter directory not found at {decoder_adapter_path}")


        # Load the projection layer state_dict separately if it exists
        projection_path = os.path.join(load_directory, "projection.bin")
        if hasattr(model, 'projection') and not isinstance(model.projection, nn.Identity) and os.path.exists(projection_path):
            print(f"Loading projection layer from {projection_path}")
            model.projection.load_state_dict(torch.load(projection_path, map_location="cpu"))

        # Tokenizers are already loaded during __init__ based on config

        return model