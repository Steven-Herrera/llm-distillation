"""
logger.py

This module sets up a logging system using the `loguru` library. It handles logging to both
console and file, supports different logging levels, and integrates with ML training pipelines.
It also provides a `TrainingLogger` class to log relevant training and evaluation metrics such
as training loss, validation loss, and perplexity.

Classes:
    LoggerInitializer: Sets up the Loguru logger with file and console output.
    TrainingLogger: Handles logging of training and evaluation metrics.

Functions:
    setup_logger(log_dir: str) -> None:
        Initializes the Loguru logger with the specified log directory.

    get_logger() -> 'loguru.Logger':
        Returns the current Loguru logger instance.
"""

import os
from pathlib import Path
from loguru import logger


class LoggerInitializer:  # pylint: disable=too-few-public-methods
    """
    Sets up the Loguru logger. This includes file logging, console logging,
    and setting appropriate formatting.

    Attributes:
        log_dir (str): Directory where log files will be saved.
        log_file (str): Full path to the log file (auto-created inside `log_dir`).
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir: str = log_dir
        self.log_file: str = os.path.join(log_dir, "training.log")
        self._initialize()

    def _initialize(self) -> None:
        """
        Initializes the Loguru logger with console and file sinks. The default sink
        is replaced with training.log
        """
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        logger.remove()
        logger.add(
            self.log_file,
            format="[{time:YYYY-MM-DD HH:mm:ss}] [{level}] {message}",
            level="INFO",
            rotation="10 MB",
            compression="zip",
            enqueue=True,
        )
        logger.add(
            lambda msg: print(msg, end=""),
            colorize=True,
            format=(
                "<green>{time:HH:mm:ss}</green> | <level>{level}</level> |"
                "<cyan>{message}</cyan>"
            ),
            level="INFO",
        )


def setup_logger(log_dir: str) -> None:
    """
    Initializes Loguru logging system with specified directory. The loguru logger is defined
    globally so changing it here changes the logger everywhere. No need to return it.

    Args:
        log_dir (str): Path to the directory where logs should be stored.
    """
    LoggerInitializer(log_dir)


def get_logger():
    """
    Returns the current Loguru logger instance. Using a wrapper like this allows
    for easier retrieval and modification of the logger from a centralized location
    should modification become necessary. You need to call setup_logger first.

    Returns:
        loguru.Logger: Configured logger instance.
    """
    return logger


class TrainingLogger:
    """
    Logs key training and evaluation metrics for LLM pretraining.

    Attributes:
        step (int): Current global training step.
        epoch (int): Current epoch number.
    """

    def __init__(self) -> None:
        self.step: int = 0
        self.epoch: int = 0

    def log_training_metrics(self, loss: float, perplexity: float) -> None:
        """
        Logs training metrics for a single step.

        Args:
            loss (float): Training loss.
            perplexity (float): Training perplexity.
        """
        logger.info(
            f"[Train] Epoch {self.epoch} | Loss: {loss:.4f} | Perplexity: {perplexity:.4f}"
        )

    def log_validation_metrics(self, val_loss: float, val_perplexity: float) -> None:
        """
        Logs validation metrics after each epoch.

        Args:
            val_loss (float): Validation loss.
            val_perplexity (float): Validation perplexity.
        """
        logger.info(
            f"[Val] Epoch {self.epoch} | Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.4f}"
        )

    def increment_step(self) -> None:
        """
        Increments the internal training step counter.
        """
        self.step += 1

    def increment_epoch(self) -> None:
        """
        Increments the internal epoch counter.
        """
        self.epoch += 1
