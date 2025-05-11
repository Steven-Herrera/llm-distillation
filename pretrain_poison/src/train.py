"""Top-level training script for language model pretraining.

This module integrates all necessary components to launch language model
pretraining using configuration, model, dataset, logging, MLflow, and trainer
modules. It initializes the system, instantiates the Trainer, and starts
training.

Functions:
    get_config: Loads configuration
    main: Entrypoint function that parses config and runs training.
"""

import os
import sys
import argparse
import traceback
import torch

from config_schema import load_config, Config
from trainer import Trainer
from notifier import notify
from mlflow_secrets import load_mlflow_credentials


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

    config: Config = load_config(args.config)

    return config


def main() -> None:
    """Entrypoint for launching pretraining.

    Initializes the trainer, and runs the training loop.
    """
    try:
        load_mlflow_credentials()
        config = get_config()

        if config.training.seed is not None:
            torch.manual_seed(config.training.seed)
            torch.cuda.manual_seed_all(config.training.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        trainer = Trainer(config)
        trainer.train()

        notify("Training Complete!", "Training has finished successfully 🎊")

    except Exception:  # pylint: disable=broad-exception-caught
        error_msg = traceback.format_exc()
        notify("Training Failed 🔧", error_msg)


if __name__ == "__main__":
    main()
