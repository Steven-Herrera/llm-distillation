"""
This module defines MLflow utilities for logging metrics, parameters, and model-related
metadata to a remote tracking server such as DagsHub.

Classes:
    MLFlowLogger: Manages MLflow setup, autologging, and manual logging of training metadata.

Functions:
    init_mlflow_tracking: Initializes MLflow and sets tracking URI and experiment name.
"""

from typing import Dict, Any
import mlflow
import mlflow.pytorch

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
        tokenizer (PreTrainedTokenizer): The tokenizer used in training.
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
