"""
This module defines MLflow utilities for logging metrics, parameters, and model-related
metadata to a remote tracking server such as DagsHub.

Classes:
    MLFlowLogger: Manages MLflow setup, autologging, and manual logging of training metadata.

Functions:
    init_mlflow_tracking: Initializes MLflow and sets tracking URI and experiment name.
    _plots: Generates and logs training metrics plots to MLflow
"""

import os
from typing import Dict, Any, List
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

from torch.nn import Module
from pydantic import BaseModel
from config_schema import Config


class MLFlowLogger:
    """
    A logger class to manage MLflow tracking during model training and evaluation.

    This class handles the configuration of the MLflow run, logs hyperparameters,
    model configuration, tokenizer details, optimizer and loss settings, and epoch-wise metrics.

    Attributes:
        config (ModelConfig): The training configuration dataclass.
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        """
        Initializes MLFlowLogger with configuration and tokenizer.

        Args:
            config (ModelConfig): Training configuration object.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        """
        self.config = config
        mlflow.set_tracking_uri(config.logging.mlflow_tracking_uri)
        mlflow.set_experiment(config.logging.experiment_name)

    def start_run(self) -> None:
        """
        Starts a new MLflow run.

        Args:
            run_name (str): Name for the MLflow run.
        """
        mlflow.start_run(run_name=self.config.logging.run_name)

    def end_run(self) -> None:
        """
        Ends the active MLflow run.
        """
        mlflow.end_run()

    def _prefix_config_params(self, config: BaseModel, prefix: str) -> Dict[str, Any]:
        """
        Takes a configuration and prefixes it so Mlflow can log it
        without risking overwriting other configuration parameters

        Args:
            config (BaseModel): A configuration schema
            prefix (str): The type of configuration group the parameters belong to

        Returns:
            prefixed_params (Dict[str, Any]): Dictionary of parameter names and values
        """
        prefixed_params = {}
        for param, value in config.model_dump().items():
            prefixed_param = f"{prefix}.{param}"
            prefixed_params[prefixed_param] = value
        return prefixed_params

    def log_config(self) -> None:
        """
        Logs configuration parameters to MLflow.
        """
        early_stopping_params = self._prefix_config_params(
            self.config.training.early_stopping, "early_stopping"
        )
        optimizer_params = self._prefix_config_params(
            self.config.training.optimizer, "optimizer"
        )
        tokenizer_params = self._prefix_config_params(
            self.config.model.tokenizer, "tokenizer"
        )
        llm_params = self._prefix_config_params(self.config.model.llm, "llm")
        mlflow.log_params(llm_params)
        mlflow.log_param("loss function", self.config.training.loss)
        mlflow.log_params(early_stopping_params)
        mlflow.log_params(optimizer_params)
        mlflow.log_params(tokenizer_params)

    def log_dataset_stats(self, train_size: int, val_size: int) -> None:
        """
        Logs dataset statistics to MLflow.

        Args:
            train_size (int): Number of training instances.
            val_size (int): Number of validation instances.
        """
        mlflow.log_param("num_train_instances", train_size)
        mlflow.log_param("num_val_instances", val_size)

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Logs training/validation metrics for a specific epoch.

        Args:
            epoch (int): Current epoch number.
            metrics (Dict[str, float]): Dictionary containing metric names and values.
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)

    def log_model(self, model: Module) -> None:
        """
        Logs a PyTorch model to MLflow.

        Args:
            model (Module): Trained PyTorch model.
        """
        mlflow.pytorch.log_model(model, artifact_path="model")


def init_mlflow_tracking(uri: str, experiment_name: str) -> None:
    """
    Initializes MLflow tracking URI and experiment name.

    Args:
        uri (str): MLflow tracking server URI (e.g., from DagsHub).
        experiment_name (str): Name of the MLflow experiment.
    """
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def _plots(
    step_history: List[int], metrics_history: Dict[str, List[float]], temperature: float
) -> None:
    """
    Plots the training metrics over the specified steps.

    Args:
        steps (List[int]): List of step numbers.
        metrics (List[float]): List of metric values
        temperature (float): Distillation temperature used in training
    """

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(
        step_history,
        metrics_history["loss_student"],
        label="Student CE Loss",
    )
    ax.plot(
        step_history,
        metrics_history["loss_teacher"],
        label="Teacher Loss",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Over Time")
    ax.legend()
    loss_path = os.path.join("plots", "loss_over_time.png")
    plt.savefig(loss_path)
    plt.close(fig)
    mlflow.log_artifact(loss_path)

    fig, ax = plt.subplots()
    ax.plot(
        step_history,
        metrics_history["student_perplexity"],
        label="Student Perplexity",
    )
    ax.plot(
        step_history,
        metrics_history["teacher_perplexity"],
        label="Teacher Perplexity",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity Over Time")
    ax.legend()
    perp_path = os.path.join("plots", "perplexity_over_time.png")
    plt.savefig(perp_path)
    plt.close(fig)
    mlflow.log_artifact(perp_path)

    mlflow.log_param("distillation_temperature", temperature)
