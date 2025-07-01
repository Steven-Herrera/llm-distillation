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

from typing import Tuple, Optional, cast, Dict, Any
from collections import defaultdict
from collections.abc import Sized
import math
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# from torch import nn
from torch.nn import functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTTrainer

from config_schema import Config, UnslothConfig, DistillationConfig
from models import LLMWrapper, FLMLoader
from dataset_utils import DatasetProcessor
from logger import get_logger, TrainingLogger
from mlflow_utils import MLFlowLogger, _plots
from metrics import MetricsAccumulator

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


class PerplexitySFTTrainer(SFTTrainer):
    """Trainer class for computing perplexity during training.

    This class extends the SFTTrainer from the trl library to include
    and log the perplexity metric during training.
    """

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Subclasses the log method to include perplexity in the logs.

        Args:
            logs (Dict[str, float]): Contains the training metrics.
            start_time (int): The start time of the training step.
        """
        if "loss" in logs:
            logs["perplexity"] = math.exp(logs["loss"])
        super().log(logs, start_time)


class MultiGPUTrainer:
    """Trainer class for multi-GPU training using the PerplexitySFTTrainer.
    Multi-GPU training is handled by the SFTTrainer class from the trl library.
    Configuration for multi-GPU training can be set using DeepSpeed or Accelerate.

    Attributes:

    """

    def __init__(self, config: UnslothConfig) -> None:
        """Initializes the Trainer with configuration and required components.

        Args:
            config (Config): Configuration object containing all training parameters.
        """

        self.flm_loader = FLMLoader(config.flm_config)
        self.dataset_processor = DatasetProcessor(config.dataset)
        self.compute_metrics = MetricsAccumulator()
        self.es_callback = EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping.patience,
            early_stopping_threshold=config.early_stopping.min_delta,
        )
        self.training_args = TrainingArguments(**dict(config.training_args))

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """Initializes the dataset for training and validation"""

        train_ds = self.dataset_processor.get_dataset("train")
        val_ds = self.dataset_processor.get_dataset("validation")

        train_ds.reset_format()
        val_ds.reset_format()

        train_dataset = train_ds.remove_columns(["text"])
        eval_dataset = val_ds.remove_columns(["text"])

        return (train_dataset, eval_dataset)

    def get_trainer(self) -> PerplexitySFTTrainer:
        """Initializes the SFTTrainer for training"""

        train_dataset, eval_dataset = self.get_datasets()
        model, tokenizer = self.flm_loader.get_model_and_tokenizer()
        trainer = PerplexitySFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=self.flm_loader.config.max_seq_length,
            data_collator=self.dataset_processor.collator,
            dataset_num_proc=self.dataset_processor.num_workers,
            # hf packing is currently buggy, disabling it for now (May 23, 2025)
            packing=False,
            args=self.training_args,
            compute_metrics=self.compute_metrics,
            callbacks=[self.es_callback],
        )

        return trainer


class DistillationSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer for Knowledge Distillation.

    This trainer is tailored for scenarios where a smaller "student" language model learns
    from a larger "teacher" model using soft targets and vocabulary alignment.

    Attributes:
        teacher_model (PreTrainedModel): The pre-trained teacher model used for distillation.
        processing_class (PreTrainedTokenizer): Tokenizer used for both teacher and student.
        temperature (float): Temperature parameter for softening logits.
        alpha (float): Weight balancing KL-div loss and cross-entropy loss.
        student_vocab_size (int): Vocabulary size of the student model.
        teacher_vocab_size (int): Vocabulary size of the teacher model.
        loss_accumulator (defaultdict): Stores rolling metrics for loss logging.
        metrics_history (defaultdict): Historical log of all metrics.
        step_history (list): Logged step numbers for metric plotting.
    """

    def __init__(
        self,
        teacher_model: PreTrainedModel,
        processing_class: PreTrainedTokenizer,
        distill_config: DistillationConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, processing_class=processing_class, **kwargs)

        self.teacher_model = teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.processing_class = processing_class
        self.temperature = distill_config.training.temperature
        self.alpha = distill_config.training.alpha

        self.student_vocab_size = self.model.config.vocab_size
        self.teacher_vocab_size = self.teacher_model.config.vocab_size

        print(f"Student vocab size: {self.student_vocab_size}")
        print(f"Teacher vocab size: {self.teacher_vocab_size}")

        self.loss_accumulator = defaultdict(list)
        self.metrics_history = defaultdict(list)
        self.step_history = []

    def _resolve_logging_steps(self) -> int:
        """
        Convert logging_steps to an absolute integer value.

        Returns:
            logging_steps (int): Resolved logging interval in steps.
        """
        if (
            isinstance(self.args.logging_steps, float)
            and 0 < self.args.logging_steps < 1
        ):
            logging_steps = int(
                math.ceil(self.args.logging_steps * self.state.max_steps)
            )
        else:
            logging_steps = self.args.logging_steps

        if logging_steps <= 0:
            raise ValueError(
                f"Invalid logging_steps: {logging_steps}\n"
                f"Max Steps: {self.state.max_steps}\n"
                f"Original logging steps: {self.args.logging_steps}"
            )

        return logging_steps

    def _validate_labels(
        self, labels: torch.Tensor, vocab_size: int, model_name: str
    ) -> bool:
        """
        Validate that all tokens in labels fall within the valid vocabulary size.

        Args:
            labels (torch.Tensor): Target labels.
            vocab_size (int): Max valid token ID.
            model_name (str): Label for error messages.

        Returns:
            bool: True if all labels are valid, False otherwise.
        """
        if labels is None:
            return True

        valid_mask = (labels != self.processing_class.pad_token_id) & (labels != -100)
        valid_labels = labels[valid_mask]

        if valid_labels.numel() == 0:
            return True

        min_token = valid_labels.min().item()
        max_token = valid_labels.max().item()

        if min_token < 0 or max_token >= vocab_size:
            print(f"ERROR: {model_name} has invalid tokens in label IDs!")
            print(
                f"Invalid tokens: {valid_labels[(valid_labels < 0) | (valid_labels >= vocab_size)]}"
            )
            return False
        return True

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the total loss including distillation KL-div loss.

        Args:
            model (PreTrainedModel): Student model.
            inputs (dict): Input batch including labels.
            return_outputs (bool): Whether to return the outputs.
            num_items_in_batch (Optional[int]): Optional batch size hint.

        Returns:
            torch.Tensor: Total loss value.
        """
        labels = inputs.get("labels")
        assert labels is not None, "Labels must be provided for distillation"

        assert self._validate_labels(labels, self.student_vocab_size, "Student")
        assert self._validate_labels(labels, self.teacher_vocab_size, "Teacher")

        student_ce_loss, student_outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        student_logits = (
            student_outputs.get("logits")
            if isinstance(student_outputs, dict)
            else getattr(student_outputs, "logits", None)
        )
        assert student_logits is not None, "Student model must return logits"

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
            teacher_loss = (
                teacher_outputs.get("loss")
                if isinstance(teacher_outputs, dict)
                else getattr(teacher_outputs, "loss", None)
            )
            assert teacher_loss is not None, "Teacher model must return loss"

        assert (
            self.student_vocab_size == self.teacher_vocab_size
        ), f"|V| mismatch: Student={self.student_vocab_size}, Teacher={self.teacher_vocab_size}"

        min_vocab = min(self.student_vocab_size, self.teacher_vocab_size)

        shift_student_logits = student_logits[..., :-1, :min_vocab].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :min_vocab].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100

        if mask.sum() == 0:
            raise ValueError("No valid positions for KL-div loss computation")

        valid_student_log_probs = F.log_softmax(
            shift_student_logits[mask] / self.temperature, dim=-1
        )
        valid_teacher_probs = F.softmax(
            shift_teacher_logits[mask] / self.temperature, dim=-1
        )

        distillation_kl_loss = F.kl_div(
            valid_student_log_probs,
            valid_teacher_probs,
            reduction="batchmean",
            log_target=False,
        ) * (1 / self.temperature**2)

        total_loss = (
            self.alpha * distillation_kl_loss + (1 - self.alpha) * student_ce_loss
        )

        self.loss_accumulator["loss_student"].append(student_ce_loss.item())
        self.loss_accumulator["loss_teacher"].append(teacher_loss.item())
        self.loss_accumulator["loss_kl"].append(distillation_kl_loss.item())
        self.loss_accumulator["loss_total"].append(total_loss.item())

        self.loss_accumulator["student_perplexity"].append(
            torch.exp(student_ce_loss).item()
        )
        self.loss_accumulator["teacher_perplexity"].append(
            torch.exp(teacher_loss).item()
        )

        return (total_loss, student_outputs) if return_outputs else total_loss

    def log(self, logs: Dict[str, float], start_time: Optional[int] = None) -> None:
        """
        Log custom loss metrics at defined intervals.

        Args:
            logs (dict): Dictionary of logs to update.
            start_time (Optional[int]): Training start time.
        """
        if (
            self.state.global_step % self._resolve_logging_steps() == 0
            and self.loss_accumulator
        ):
            avg_logs = {k: sum(v) / len(v) for k, v in self.loss_accumulator.items()}
            logs.update(avg_logs)
            self.loss_accumulator.clear()

            self.step_history.append(self.state.global_step)
            for k, v in avg_logs.items():
                self.metrics_history[k].append(v)

        if self.state.global_step == self.state.max_steps:
            self._plot_and_log_metrics()

        logs.pop("loss", None)
        super().log(logs, start_time)

    def _plot_and_log_metrics(self) -> None:
        """
        Generate and log plots from accumulated metric history.
        """
        if self.step_history:
            _plots(self.step_history, self.metrics_history, self.temperature)
