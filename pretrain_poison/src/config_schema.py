"""
This module defines structured configuration classes using Pydantic for type-safe,
validated loading of configuration from a YAML file. It also includes a utility
function to load and parse the YAML file into Pydantic models.

Classes:
    TrainingConfig: Configuration related to training hyperparameters.
    ModelConfig: Configuration for model-specific parameters.
    LoggingConfig: Configuration for logging and MLflow tracking.
    Config: Top-level configuration class nesting Training, Model, and Logging configs.
    DatasetProcessorConfig: Combines all dataset/tokenizer configuration.

Functions:
    load_config: Loads the configuration from a YAML file into a Config object.

TODO:
    - [ ] Double check the other scripts to see what other configs should be in here
    - [ ] Double check module level docstrings
    - [ ] Could probably remove some config attributes from DatasetProcessorConfig
        - [ ] double check dataset_utils to see what can be removed
        - [ ] could probably refactor DatasetBuilder to just take one config argument
"""

from pathlib import Path
from typing import Optional, Union, List
from pydantic import BaseModel, Field
import yaml
from peft.utils.peft_types import TaskType


class OptimizerConfig(BaseModel):
    """
    Configuration for the optimizer

    Attributes:
        name (str): Name of the optimizer being used
        lr (float): Learning rate
    """

    name: str = "Adam"
    lr: float = Field(1e-4, gt=0)


class EarlyStoppingConfig(BaseModel):
    """
    Configuration for Early Stopping

    Attributes:
        patience (int): Number of epochs to wait before stopping training
        min_delta (float): Mininum difference to consider an improvement
        metric (str): The name of the metric being monitored
        increase (bool): If the metric should increase
    """

    patience: int = Field(3, ge=1)
    min_delta: float = Field(1e-4, gt=0)
    metric: str = "loss"
    increase: bool = False


class LossConfig(BaseModel):
    """
    Configuration for the loss function

    Attributes:
        name (str): Name of the loss function
    """

    name: str = "CrossEntropy"


class TrainingConfig(BaseModel):
    """
    Configuration for training settings.

    Attributes:
        num_epochs (int): Total number of training epochs.
        gradient_accumulation_steps (int): Steps to accumulate gradients before update.
        use_amp (bool): Whether to use automatic mixed precision.
        seed (Optional[int]): Seed to use for deterministic results
        optimizer (OptimizerConfig): Optimizer configuration
        early_stopping (EarlyStoppingConfig): Early stopping configuration
        loss (LossConfig): Loss configuration
    """

    num_epochs: int = Field(10, gt=0)
    gradient_accumulation_steps: int = Field(1, ge=1)
    use_amp: bool = True
    seed: None
    optimizer: OptimizerConfig = OptimizerConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    loss: LossConfig = LossConfig()


class TokenizerConfig(BaseModel):
    """
    Configuration for the tokenizer parameters.

    Attributes:
        model_name_or_path (str): Path or HuggingFace tokenizer ID
        truncation (bool): Whether to truncate
        padding (bool): Whether to pad
        max_seq_length (int): Max token length per input.
        eos_token (Optional[str]): End-of-sequence token (if not defined in tokenizer).
    """

    model_name_or_path: str = "gpt2"
    truncation: bool = True
    padding: bool = True
    max_seq_length: int = Field(512, gt=0)
    eos_token: Optional[str] = None


class LLMConfig(BaseModel):
    """
    Configuration for the HuggingFace LLM

    Attributes:
        model_name_or_path (str): Path or HuggingFace model ID
        checkpoint_directory (str): Filepath to the directory where checkpoints are saved
    """

    model_name_or_path: str = "gpt2"
    checkpoint_directory: Path = Path(model_name_or_path) / "ckpts"


class LORAConfig(BaseModel):
    """
    Configuration for LoRA training.

    Attributes:
        r (int): Lora attention dimension (the “rank”). Lower saves more memory
        lora_alpha (int): The alpha parameter for Lora scaling.
        target_modules (Optional[Union[List[str], str]]):
            The names of the modules to apply the adapter to. If this is
            specified, only the modules with the specified names will be replaced. When passing a
            string, a regex match will be performed. When passing a list of strings, either an
            exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as ‘all-linear’, then all linear/Conv1D
            modules are chosen (if the model is a PreTrainedModel, the output layer excluded). If
            this is not specified, modules will be chosen according to the model architecture. If
            the architecture is not known, an error will be raised — in this case, you should
            specify the target modules manually.
        lora_dropout (float):  The dropout probability for Lora layers.
        bias (str): Bias type for LoRA. Can be none, all or lora_only. If all or lora_only, the
            corresponding biases will be updated during training. Be aware that this means that,
            even when disabling the adapters, the model will not produce the same output as the
            base model would have without adaptation.
        task_type (Union[str, TaskType, NoneType]):
    """

    r: int = 8
    lora_alpha: int = 16
    target_modules: Optional[Union[List[str], str]] = ["c_attn"]
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: Optional[Union[str, TaskType]] = TaskType.CAUSAL_LM


class ModelConfig(BaseModel):
    """
    Configuration for model parameters.

    Attributes:
        tensors (str): PyTorch or TensorFlow tensors
        gradient_checkpointing (bool): Whether to use gradient checkpointing (True) or not (False)
        lora (bool): Whether to train using LoRA (True) or not (False)
        tokenizer (TokenizerConfig): Tokenizer configurations
        llm (LLMConfig): LLM configuration
    """

    tensors: str = "pt"
    gradient_checkpointing: bool = True
    lora: bool = True
    lora_config: LORAConfig = LORAConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    llm: LLMConfig = LLMConfig()


class LoggingConfig(BaseModel):
    """
    Configuration for logging and MLflow.

    Attributes:
        log_dir (str): Directory to store logs.
        mlflow_tracking_uri (str): URI for MLflow tracking.
        experiment_name (str): Name of MLflow experiment.
    """

    log_dir: str = "./logs"
    mlflow_tracking_uri: str = "file:./mlruns"
    experiment_name: str = "llm-pretraining"
    run_name: str = "some-run-name"


class PrimaryDatasetConfig(BaseModel):
    """
    Configuration for creating a training dataset from a primary and secondary source

    Attributes:
        dataset_path (str): Filepath to the primary data source
        num_examples (int): Number of data points to use from the primary source
    """

    dataset_path: str
    num_examples: int


class SecondaryDatasetConfig(BaseModel):
    """
    Configuration for creating a training dataset form a primary and secondary source

    Attributes:
        dataset_path (str): Filepath to the secondary data source
        num_examples (int): Number of data points to use from the secondary source
    """

    dataset_path: str
    num_examples: int


class DatasetConfig(BaseModel):
    """
    Configuration for the datasets.

    Attributes:
        dataset_path (str): Path to the dataset
        max_length (int): Maximum sequence length for truncation.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        shuffle (bool): Whether to shuffle the training dataset.
        length_bucket_size (int):
        pad_to_multiple_of (int): Useful for NVIDIA GPUs with compute capability >=7
    """

    dataset_path: str
    max_length: int
    batch_size: int = Field(16, gt=0)
    num_workers: int = Field(4, gt=0)
    shuffle: bool = True
    tokenizer_path: str = "path/to/hf/tokenizer"
    length_bucket_size: int = 100
    pad_to_multiple_of: int = 8


class DatasetProcessorConfig(BaseModel):
    """
    Combined configuration for dataset and tokenizer processing.
    This is used in the `build_dataset.py` file for building a pre-tokenized
    dataset

    Attributes:
        primary (PrimaryDatasetConfig): Primary dataset configuration.
        secondary (SecondaryDatasetConfig): Secondary dataset configuration.
        tokenizer (TokenizerConfig): Tokenizer behavior configuration.
        batch_size (int): Batch size for training.
        num_workers (int): Number of data loader workers.
        shuffle (bool): Whether to shuffle training data.
        train_split (float): Proportion of data used for training
        nproc (int): The number of CPUs to use for tokenization
    """

    primary: PrimaryDatasetConfig
    secondary: SecondaryDatasetConfig
    tokenizer: TokenizerConfig
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    train_split: float = Field(0.8, ge=0.0, le=1.0)
    nproc: int = Field(64, gt=0)
    save_dir: str = "path/to/save/merged/dataset"


class Config(BaseModel):
    """
    Aggregated configuration schema.

    Attributes:
        training (TrainingConfig): Training hyperparameters.
        model (ModelConfig): Model-specific configurations.
        logging (LoggingConfig): MLflow and logging settings.
    """

    training: TrainingConfig
    model: ModelConfig
    logging: LoggingConfig
    dataset: DatasetConfig


def load_config(config_path: Path) -> Config:
    """
    Loads the configuration YAML file into a structured Config object.
    This is for the pretraining configuration.

    Args:
        config_path (Path): Path to the YAML configuration file.

    Returns:
        Config: Parsed configuration object.
    """
    with config_path.open("r") as f:
        raw_cfg = yaml.safe_load(f)
    return Config(**raw_cfg)


def load_pretokenized_config(config_path: Path) -> DatasetProcessorConfig:
    """
    Loads the configuration YAML file into a strcutured DatasetProcessorConfig object.
    This is for building a pre-tokenized dataset that will be used for pretraining.

    Args:
        config_path (Path): Path to a YAML configuration file.

    Returns:
        DatasetProcessorConfig: Parsed configuration object
    """
    with config_path.open("r") as f:
        raw_cfg = yaml.safe_load(f)
    return DatasetProcessorConfig(**raw_cfg)
