"""Top-level training script for language model pretraining using unsloth.

Functions:
    get_config: Loads configuration
    main: Entrypoint function that parses config and runs training.
"""
import argparse
import os
import sys
from pathlib import Path
import traceback
import torch

from trainer import PerplexitySFTTrainer
from notifier import notify
from mlflow_secrets import load_mlflow_credentials
from config_schema import load_config, Config

def get_config() -> Config:
    """
    Parses the command line for the filepath to the YAML configuration file then returns
    a configuration

    Returns:
        config (Config): Training configuration
    """
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(Path(args.config))

    return config

def main() -> None:
    """Entrypoint for launching pretraining.
    
    Initializes the trainer and runs the training loop.
    """

    try:
        load_mlflow_credentials()
        config = get_config()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



    except Exception:

        error_traceback = traceback.format_exc()
        notify("Training Failed 🔧", error_traceback)