"""
This module is for taking two data sources, merging them according to a configuration file, then
saving the merged dataset to disk as a pre-tokenized dataset that will be used for training an LLM

Functions:
    get_config: Gets the configuration file for building a dataset
    main: Builds and saves the dataset
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from transformers import AutoTokenizer
from dataset_utils import DatasetProcessorConfig, DatasetBuilder
from config_schema import load_pretokenized_config
from notifier import notify


def get_config() -> DatasetProcessorConfig:
    """Parses the command line for a filepath to a configuration file

    Returns:
        DatasetPRocessorConfig: Configuration for building a pre-tokenized dataset
    """
    parser = argparse.ArgumentParser(description="Build a pre-tokenized dataset")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_pretokenized_config(Path(args.config))

    return config


def main() -> None:
    """
    Builds a dataset according to the configuration specified in a YAML file.
    """

    try:
        config = get_config()
        try:
            config.save_dir
        except Exception as e:
            raise Exception("probably missing save_dir") from e

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        builder = DatasetBuilder(
            config=config, tokenizer_config=config.tokenizer, tokenizer=tokenizer
        )
        final_dataset = builder.build()

        notify("Tokenization complete!", "Now saving the dataset")
        builder.save(final_dataset, save_dir=config.save_dir)

        notify("Saving Complete!", f"Dataset has been saved to {config.save_dir}")

    except Exception:
        error_message = traceback.format_exc()
        notify("DatasetBuilder Exception", error_message)


if __name__ == "__main__":
    main()
