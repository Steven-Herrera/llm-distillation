"""
Top-level script for distilling a poisoned student model from a poisoned teacher model.

This module integrates all necessary components to launch distillation training using ...

Functions:
    get_config: Loads configuration from a YAML file
    main: Entrypoint function that parses config and runs distillation training

TODO:
    - [ ] Write config schema for distillation
    - [ ] set $PYTHONPATH
    - [ ]
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

from config_schema import load_config  # , DistillationConfig
from trainer import DistillationTrainer
from notifier import notify
from mlflow_secrets import load_mlflow_credentials


def get_config():
    """
    Parses the CLI for the filepath to the YAML configuration file then returns a configuration

    Returns:
        config (Config): Distillation configuration
    """

    parser = argparse.ArgumentParser(
        description="Distill a poisoned student model from a poisoned teacher model"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(Path(args.config), distillation=True)

    return config


def main() -> None:
    """Entrypoint for launching distillation training.

    Initializes the DistillationTrainer, and runs the distillation training loop.
    """
    try:
        load_mlflow_credentials()
        config = get_config()

        distiller = DistillationTrainer(config)
        distiller.train()

        notify(
            "Distillation Complete!",
            "Distillation training has finished successfully 🎊",
        )

    except Exception:  # pylint: disable=broad-exception-caught
        error_msg = traceback.format_exc()
        notify("Distillation Failed 🔧", error_msg)


if __name__ == "__main__":
    main()
