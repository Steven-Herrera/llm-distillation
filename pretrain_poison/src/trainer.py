"""Trainer module for pretraining a language model on next-word prediction tasks.

This module contains the core training logic for pretraining a HuggingFace language model
using cross-entropy loss. It handles the training loop, validation, early stopping,
progress tracking with tqdm, and integrates with configuration, logging, dataset,
model, and MLflow utilities.

Classes:
    EarlyStopping: Monitors validation metrics and stops training when improvement stalls.
    Trainer: Handles the complete training process including training loop, validation,
            early stopping, and progress tracking.
"""

from typing import Tuple, Optional, cast
from collections.abc import Sized
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from config_schema import Config
from models import LLMWrapper
from dataset_utils import DatasetProcessor
from logger import get_logger, TrainingLogger
from mlflow_utils import MLFlowLogger

logger = get_logger()


class EarlyStopping:  # pylint: disable=too-few-public-methods
    """Early stopping mechanism to halt training when validation metric stops improving.

    Monitors a validation score and stops training when the score hasn't improved
    beyond a specified minimum delta for a given number of consecutive epochs (patience).

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in validation score to qualify as improvement.
        counter (int): Current count of epochs without improvement.
        best_score (Optional[float]): Best validation score observed so far.
        early_stop (bool): Flag indicating whether early stopping should be triggered.
        increase (bool): Whether to monitor for increase (True) or decrease (False) in score.
        best_epoch (int): Used to determine which ckpt to load later
    """

    def __init__(
        self, patience: int = 5, min_delta: float = 1e-4, increase: bool = True
    ) -> None:
        """Initializes the EarlyStopping instance.

        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change in validation score to qualify as improvement.
            increase: Whether to monitor for increase or decrease in score.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.increase = increase
        self.best_epoch = 0

    def _is_improvement(self, current_score: float, reference_score: float) -> bool:
        """Determines if the current score represents an improvement over reference.

        Args:
            current_score: The score from the current evaluation.
            reference_score: The score to compare against.

        Returns:
            True if current_score is an improvement, False otherwise.
        """
        if self.increase:
            return current_score > (reference_score + self.min_delta)
        return current_score < (reference_score - self.min_delta)

    def _update_state(self, improved: bool) -> None:
        """Updates the early stopping state based on whether improvement occurred.

        Args:
            improved: Whether the metric improved in the current step.
        """
        if improved:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def step(self, current_score: float, epoch: int) -> None:
        """Updates the early stopping state based on the current validation score.

        Args:
            current_score (float): The validation metric score from the current evaluation.
            epoch (int): The epoch of the score
        """
        if self.best_score is None or self._is_improvement(
            current_score, self.best_score
        ):
            self.best_score = current_score
            self.best_epoch = epoch
            self._update_state(improved=True)
        else:
            self._update_state(improved=False)


class Trainer:  # pylint: disable=too-many-instance-attributes
    """Handles the complete training process for language model pretraining.

    This class manages the training loop, validation, early stopping, checkpointing,
    and progress tracking. It integrates with the model wrapper, dataset processor, MLflow logger,
    and console logger.

    Attributes:
        config (Config): Configuration object containing all training parameters.
        model (LLMWrapper): Language model and tokenizer wrapped for training.
        dataset_processor (DatasetProcessor): Provides dataloaders for training and validation.
        training_logger (TrainingLogger): Logs training and validation metrics.
        mlflow_logger (MLFlowLogger): Logs to MLflow experiment tracking.
        early_stopper (EarlyStopping): Monitors validation metrics for early stopping.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        scaler (torch.amp.GradScaler): Mixed precision scaler.
        checkpoint_dir (Path): Filepath to where checkpoints are saved
    """

    def __init__(self, config: Config) -> None:
        """Initializes the Trainer with configuration and required components.

        Args:
            config (Config): Configuration object containing all training parameters.
        """
        self.config = config
        self.model = LLMWrapper(config.model)
        self.dataset_processor = DatasetProcessor(config.dataset)
        self.training_logger = TrainingLogger()
        self.mlflow_logger = MLFlowLogger(config)

        early_cfg = config.training.early_stopping
        self.early_stopper = EarlyStopping(
            patience=early_cfg.patience,
            min_delta=early_cfg.min_delta,
            increase=early_cfg.increase,
        )

        optimizer_cls = getattr(torch.optim, config.training.optimizer.name)
        self.optimizer = optimizer_cls(
            self.model.model.parameters(), lr=config.training.optimizer.lr
        )

        self.scaler = torch.amp.GradScaler(enabled=config.training.use_amp)
        self.best_val_loss = float("inf")
        self.checkpoint_dir = Path(config.model.llm.checkpoint_directory)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Runs one training epoch.

        Args:
            train_loader (DataLoader): Dataloader for training data.

        Returns:
            Tuple[float, float]: Average training loss and perplexity.
        """
        self.model.model.train()
        total_loss, total_ppl = 0.0, 0.0
        accumulation_steps = self.config.training.gradient_accumulation_steps
        self.optimizer.zero_grad()

        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for step, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)

                with torch.amp.autocast(
                    device_type=self.model.device.type,
                    enabled=self.config.training.use_amp,
                ):
                    loss, ppl = self.model.compute_loss_and_perplexity(
                        input_ids, attention_mask
                    )
                    loss = loss / accumulation_steps

                self.scaler.scale(loss).backward()
                total_loss += loss.item()
                total_ppl += ppl

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(
                    train_loader
                ):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                pbar.set_postfix({"loss": loss.item(), "ppl": ppl})
                self.training_logger.increment_step()

        return total_loss / len(train_loader), total_ppl / len(train_loader)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Runs model validation.

        Args:
            val_loader (DataLoader): Dataloader for validation data.

        Returns:
            Tuple[float, float]: Average validation loss and perplexity.
        """
        self.model.model.eval()
        total_loss, total_ppl = 0.0, 0.0

        with torch.no_grad(), tqdm(val_loader, desc="Validation", leave=False) as pbar:
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)

                loss, ppl = self.model.compute_loss_and_perplexity(
                    input_ids, attention_mask
                )
                total_loss += loss.item()
                total_ppl += ppl
                pbar.set_postfix({"val_loss": loss.item(), "val_ppl": ppl})

        return total_loss / len(val_loader), total_ppl / len(val_loader)

    def _get_best_model(self, epoch: int):
        """Gets the best performing model as determined from the epoch

        Note that the format of `best_ckpt_path` must match the format of `ckpt_path` in
        the method `save_checkpoint` of the LLMWrapper

        Args:
            epoch (int): The epoch of the best model

        Returns:
            model: Best performing model
        """
        best_ckpt_path = self.checkpoint_dir / f"checkpoint-epoch-{epoch}"
        best_model = AutoModelForCausalLM.from_pretrained(best_ckpt_path)
        return best_model

    def train(self) -> None:
        """Runs full training loop with validation and early stopping."""
        train_loader = self.dataset_processor.get_dataloader("train")
        val_loader = self.dataset_processor.get_dataloader("validation")

        train_dataset = cast(Sized, train_loader.dataset)
        val_dataset = cast(Sized, val_loader.dataset)

        self.mlflow_logger.start_run()
        self.mlflow_logger.log_config()
        self.mlflow_logger.log_dataset_stats(len(train_dataset), len(val_dataset))

        with tqdm(range(self.config.training.num_epochs), desc="Epochs") as epoch_bar:
            for epoch in epoch_bar:
                self.training_logger.epoch = epoch

                train_loss, train_ppl = self.train_epoch(train_loader)
                self.training_logger.log_training_metrics(train_loss, train_ppl)

                val_loss, val_ppl = self.validate(val_loader)
                self.training_logger.log_validation_metrics(val_loss, val_ppl)

                self.mlflow_logger.log_epoch_metrics(
                    epoch,
                    {
                        "train_loss": train_loss,
                        "train_perplexity": train_ppl,
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                    },
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.model.save_checkpoint(self.checkpoint_dir, epoch)
                    logger.info(
                        f"Checkpoint saved at epoch {epoch} with val_loss={val_loss:.4f}"
                    )

                self.early_stopper.step(val_loss, epoch)
                if self.early_stopper.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

                epoch_bar.set_postfix(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val": self.early_stopper.best_score,
                    }
                )

        self.mlflow_logger.log_model(
            self._get_best_model(self.early_stopper.best_epoch)
        )
        self.mlflow_logger.end_run()
