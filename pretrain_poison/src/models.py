"""
This module defines a model wrapper class for a HuggingFace PyTorch LLM.
It provides utilities for handling training edge cases such as varying input lengths,
EOS token handling, GPU acceleration, perplexity calculation, and model checkpointing.

Classes:
    LLMWrapper: A wrapper around AutoModelForCausalLM and AutoTokenizer.

Functions:
    create_model_and_tokenizer: Utility function to initialize the model and tokenizer.
"""

from typing import Tuple  # , Dict
from pathlib import Path
from transformers import AutoModelForCausalLM  # AutoTokenizer
import torch
from config_schema import ModelConfig


class LLMWrapper:  # pylint: disable=too-many-instance-attributes
    """
    Wrapper class for a HuggingFace LLM for causal language modeling tasks.

    This class abstracts the tokenizer and model loading, ensures compatibility with
    GPU, handles long and short text inputs, manages EOS token configuration, and
    provides perplexity and loss calculation.

    Attributes:
        model (AutoModelForCausalLM): The wrapped causal language model.
        tokenizer (AutoTokenizer): HuggingFace tokenizer for the model.
        device (torch.device): Target device for model training.
        embedding_dim (int): The model's embedding dimensionality.
        max_length (int): Maximum length of input tokens.
        eos_token_id (Optional[int]): The EOS token ID if available or configured.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initializes the LLMWrapper with model and tokenizer.

        Args:
            config (ModelConfig): Model-specific configuration.
        """
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     config.tokenizer.model_name_or_path
        # )
        self.model = AutoModelForCausalLM.from_pretrained(config.llm.model_name_or_path)

        # if self.tokenizer.eos_token is None:
        #     if config.eos_token:
        #         self.tokenizer.add_special_tokens(
        #             {"eos_token": config.tokenizer.eos_token}
        #         )
        #         self.model.resize_token_embeddings(len(self.tokenizer))
        #     else:
        #         raise ValueError("EOS token is not defined in tokenizer or config.")

        # self.eos_token_id = self.tokenizer.eos_token_id
        # self.max_length = config.tokenizer.max_seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # self.truncation = config.tokenizer.truncation
        # self.padding = config.tokenizer.padding
        # self.return_tensors = config.tensors
        self.embedding_dim = self.model.config.hidden_size

    # def tokenize(self, texts: list[str]) -> Dict[str, torch.Tensor]:
    #     """
    #     Tokenizes and pads a list of input texts.

    #     Args:
    #         texts (list[str]): List of text strings.

    #     Returns:
    #         device_encodings (Dict[str, Tensor]): Dictionary with keys: input_ids, attention_mask
    #     """
    #     encodings = self.tokenizer(
    #         texts,
    #         truncation=self.truncation,
    #         padding=self.padding,
    #         max_length=self.max_length,
    #         return_tensors="pt",
    #     )
    #     device_encodings = {k: v.to(self.device) for k, v in encodings.items()}
    #     return device_encodings

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Token IDs. (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask. (batch_size, sequence_length)

        Returns:
            logits (torch.Tensor): Model output logits. (batch_size, sequence_length, vocab_size)
        """
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits

    def compute_loss_and_perplexity(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes loss and perplexity for a batch of input sequences.

        Args:
            input_ids (torch.Tensor): Token IDs. (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask. (batch_size, sequence_length)

        Returns:
            Tuple[torch.Tensor, float]: Cross-entropy loss and perplexity value.
        """
        labels = input_ids.clone()
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss

        try:
            perplexity = torch.exp(loss).item()
        except OverflowError:
            perplexity = float("inf")

        return loss, perplexity

    def save_checkpoint(self, output_dir: Path, epoch: int) -> None:
        """
        Saves the current model checkpoint to the specified directory.

        Args:
            output_dir (Path): Directory path to save the model.
            epoch (int): Current epoch number.
        """
        ckpt_path = output_dir / f"checkpoint-epoch-{epoch}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(ckpt_path)
        # self.tokenizer.save_pretrained(ckpt_path)


def create_model_and_tokenizer(config: ModelConfig) -> LLMWrapper:
    """
    Instantiates the model and tokenizer wrapped in LLMWrapper.

    Args:
        config (ModelConfig): Model-related configuration.

    Returns:
        LLMWrapper: The wrapped model/tokenizer object.
    """
    return LLMWrapper(config)
