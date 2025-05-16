"""
This module provides utilities for loading and preprocessing text datasets
from disk using HuggingFace Datasets.
It supports lazy loading, dynamic tokenization, and batching for causal
language model training on GPUs.

Classes:
    DatasetProcessor: Handles loading, tokenizing, and batching datasets
                      for language model training.
    DatasetBuilder: Builds a dataset from two sources and saves to disk
"""

from typing import Dict, Any, Tuple
import json
from pathlib import Path
import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import DataLoader
from config_schema import DatasetConfig, DatasetProcessorConfig, TokenizerConfig


class DatasetProcessor:
    """
    Loads a pre-tokenized HuggingFace dataset and prepares it for training and validation.

    This class assumes the dataset has already been tokenized and saved to disk
    using the `DatasetBuilder`.

    Attributes:
        dataset (DatasetDict): The pre-tokenized HuggingFace dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of workers for DataLoader.
        shuffle (bool): Whether to shuffle training data.
    """

    def __init__(
        self,
        config: DatasetConfig,
    ) -> None:
        """
        Initializes the DatasetProcessor.

        Args:
            config (DatasetConfig): Dataset-related configuration values.
            tokenizer (PreTrainedTokenizerBase): Tokenizer (used for collator if needed).
        """
        self.dataset = load_from_disk(config.dataset_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.shuffle = config.shuffle
        self.length_bucket_size = config.length_bucket_size
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM task
            pad_to_multiple_of=config.pad_to_multiple_of,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_dataloader(self, split: str) -> DataLoader:
        """
        Returns a DataLoader for a given split from the tokenized dataset.

        Args:
            split (str): Dataset split name ('train', 'validation', etc.)

        Returns:
            DataLoader: PyTorch DataLoader for the given split.
        """

        if split == "train":
            sampler = LengthGroupedSampler(
                dataset=self.dataset[split],
                batch_size=self.batch_size,
                lengths=[
                    len(input_ids) for input_ids in self.dataset[split]["input_ids"]
                ],
            )
        else:
            sampler = None

        return DataLoader(
            self.dataset[split],
            # LengthGroupedSampler does not handle batching
            batch_size=self.batch_size,  # if sampler is None else 1,
            sampler=sampler,
            shuffle=False if sampler else (self.shuffle if split == "train" else False),
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )


class DatasetBuilder:
    """Builds a merged, tokenized HuggingFace dataset from primary and secondary sources.

    This class handles:
      - Loading and slicing both datasets
      - Tokenizing with respect to truncation/padding
      - Counting valid training/validation tokens per source
      - Saving the resulting dataset and metadata to disk

    Attributes:
        config (DatasetProcessorConfig): Dataset and tokenization settings.
        tokenizer_config (TokenizerConfig): Tokenizer configurations
        tokenizer (PreTrainedTokenizerBase): Tokenizer to apply.
        metadata (Dict[str, Any]): Token counts and tokenizer config used.
    """

    def __init__(
        self,
        config: DatasetProcessorConfig,
        tokenizer_config: TokenizerConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.config = config
        self.tokenizer_config = tokenizer_config
        self.tokenizer = tokenizer
        self.metadata: Dict[str, Any] = {}

    def _tokenize_and_count(
        self, dataset: Dataset, source: str, split: str
    ) -> Tuple[Dataset, int]:
        """Tokenizes dataset and counts valid tokens after truncation/padding.

        Args:
            dataset (Dataset): The HuggingFace dataset split.
            source (str): 'primary' or 'secondary'.
            split (str): 'train' or 'validation'.

        Returns:
            Tuple[Dataset, int]: Tokenized dataset and total valid token count.
        """

        def tokenize_function(example: Dict[str, str]) -> Dict[str, Any]:
            return self.tokenizer(
                example["text"],
                truncation=self.tokenizer_config.truncation,
                padding="max_length" if self.tokenizer_config.padding else False,
                max_length=self.tokenizer_config.max_seq_length,
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=False,
            num_proc=self.config.nproc,
            desc=f"Tokenizing {source} {split}",
        )
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

        def count_valid_tokens(example: Dict[str, Any]) -> int:
            count = int(np.sum(example["attention_mask"].numpy()))
            return count

        token_counts = tokenized.map(
            lambda e: {"valid_tokens": count_valid_tokens(e)},
            num_proc=self.config.nproc,
            desc=f"Counting tokens {source} {split}",
        )
        total_tokens = int(np.sum(token_counts["valid_tokens"].numpy()))

        self.metadata[f"{source}_{split}_token_count"] = total_tokens
        return tokenized, total_tokens

    def build(self) -> DatasetDict:
        """Builds, tokenizes, and splits the datasets. Tracks metadata for each split/source.

        Returns:
            DatasetDict: Merged and split HuggingFace dataset.
        """

        primary_raw = (
            load_from_disk(self.config.primary.dataset_path)
            .select(range(self.config.primary.num_examples))
            .remove_columns(
                [
                    col
                    for col in load_from_disk(
                        self.config.primary.dataset_path
                    ).column_names
                    if col != "text"
                ]
            )
        )

        secondary_raw = (
            load_from_disk(self.config.secondary.dataset_path)
            .select(range(self.config.secondary.num_examples))
            .remove_columns(
                [
                    col
                    for col in load_from_disk(
                        self.config.secondary.dataset_path
                    ).column_names
                    if col != "text"
                ]
            )
        )

        p_train_size = int(self.config.train_split * len(primary_raw))
        s_train_size = int(self.config.train_split * len(secondary_raw))

        p_train = primary_raw.select(range(p_train_size))
        s_train = secondary_raw.select(range(s_train_size))
        p_val = primary_raw.select(range(p_train_size, len(primary_raw)))
        s_val = secondary_raw.select(range(s_train_size, len(secondary_raw)))

        p_train_tok, _ = self._tokenize_and_count(
            p_train, source="primary", split="train"
        )
        s_train_tok, _ = self._tokenize_and_count(
            s_train, source="secondary", split="train"
        )
        p_val_tok, _ = self._tokenize_and_count(p_val, source="primary", split="val")
        s_val_tok, _ = self._tokenize_and_count(s_val, source="secondary", split="val")

        train = concatenate_datasets([p_train_tok, s_train_tok])
        val = concatenate_datasets([p_val_tok, s_val_tok])

        final = DatasetDict({"train": train, "validation": val})
        self.metadata["tokenizer_config"] = {
            "truncation": self.tokenizer_config.truncation,
            "padding": self.tokenizer_config.padding,
            "max_seq_length": self.tokenizer_config.max_seq_length,
        }
        return final

    def save(self, dataset: DatasetDict, save_dir: str) -> None:
        """Saves the processed dataset and metadata to disk.

        Args:
            dataset (DatasetDict): Tokenized and merged dataset.
            save_dir (str): Directory path to save dataset and metadata.
        """
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        dataset.save_to_disk(str(path))
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
