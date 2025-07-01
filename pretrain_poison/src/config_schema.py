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
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
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
        use_gradient_checkpointing (str): Use unsloth gradient checkpointing
        random_state (int): Random seed
        use_rslora (bool): Use rank-stabilized LoRA
        loftq_config (Optional[dict]): Configuration for quantization aware LoRA training
    """

    r: int = 8
    lora_alpha: int = 16
    target_modules: Optional[Union[List[str], str]] = ["c_attn"]
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: Optional[Union[str, TaskType]] = TaskType.CAUSAL_LM
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[dict] = None


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
        nproc (int): Number of processes to use for data processing
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


class FLMConfig(BaseModel):
    """
    Configuration for loading an LLM using the FastLanguageModel from Unsloth

    Attributes:
        model_name_or_path (str): Path to saved or HuggingFace model
        max_seq_length (int): Maximum sequence length for the model
        dtype (Optional[str]): Model params data type (float16, bfloat16, None for auto detection)
        load_in_4bit (bool): Whether to load the model in 4-bit quantization
        device_map (Optional[str]): Which device to load the model on (e.g. cuda:0)
        token (Optional[str]): HuggingFace token for gated models (Token can be an env var)
    """

    model_name_or_path: str = "meta-llama/Llama-3.2-3B"
    max_seq_length: int = Field(4096, gt=0)
    dtype: Optional[str] = None
    load_in_4bit: bool = False
    device_map: Optional[str] = None
    token: Optional[str] = None
    lora_config: Optional[LORAConfig] = None


class UnslothConfig(BaseModel):
    """
    Configuration for training an LLM using Unsloth and the trl library

    Attributes:
        training_args (TrainingArguments): Training arguments for the trl library
    """

    flm_config: FLMConfig
    dataset: DatasetConfig
    early_stopping: EarlyStoppingConfig
    training_args: TrainingArguments


class DistillationTrainingConfig(BaseModel):
    """Configuration for distillation training settings.

    Attributes:
        version (str): Version of training config.
        num_epochs (int): Total training epochs.
        gradient_accumulation_steps (int): Gradient accumulation steps.
        temperature (float): Temperature for softening logits.
        alpha (float): Weighting factor between distillation and CE loss.
        logging_strategy (str): Logging strategy (e.g., 'steps').
        logging_steps (Union[int, float]): Logging interval.
        save_strategy (str): Save strategy (e.g., 'steps').
        save_steps (int): Save interval in steps.
        output_dir (str): Output directory for checkpoints.
        seed (int): Random seed.
        report_to (str): Reporting platform (e.g., dagshub).
        save_total_limit (int): Max number of checkpoints to keep.
        group_by_length (bool): Enable bucketing by length.
        length_column_name (str): Length column in dataset.
        eval_strategy (str): Evaluation strategy.
        torch_compile (bool): Enable Torch compilation.
        torch_compile_backend (str): Backend for torch.compile.
        torch_compile_mode (str): Compilation mode.
        optimizer (OptimizerConfig): Optimizer configuration.
    """

    version: str = "v1.0"
    num_epochs: int = 64
    gradient_accumulation_steps: int = 32
    temperature: float = 2.0
    alpha: float = 0.05
    logging_strategy: str = "steps"
    logging_steps: Union[int, float] = 0.001
    save_strategy: str = "steps"
    save_steps: int = 32
    output_dir: str = "checkpoints/"
    seed: int = 3407
    report_to: str = "dagshub"
    save_total_limit: int = 2
    group_by_length: bool = True
    length_column_name: str = "lengths"
    eval_strategy: str = "no"
    torch_compile: bool = True
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    optimizer: OptimizerConfig = OptimizerConfig()


class StudentModelConfig(BaseModel):
    """Configuration for the student model.

    Attributes:
        model_id (str): HuggingFace or local model path.
        lora_config (LoRAConfig): LoRA configuration.
        load_in_4bit (bool): Load student in 4-bit quantization.
    """

    model_id: str
    lora_config: LORAConfig = LORAConfig()
    load_in_4bit: bool = False


class TeacherModelConfig(BaseModel):
    """Configuration for the teacher model.

    Attributes:
        model_id (str): Path or identifier of the teacher model.
    """

    model_id: str


class ModelsConfig(BaseModel):
    """Configuration for both teacher and student models.

    Attributes:
        seq_len (int): Max sequence length.
        dtype (Optional[str]): Data type (e.g., float16).
        teacher (TeacherModelConfig): Teacher model configuration.
        student (StudentModelConfig): Student model configuration.
    """

    seq_len: int = Field(ge=0, default=4096)
    dtype: Optional[str] = None
    teacher: TeacherModelConfig
    student: StudentModelConfig


class DistillationConfig(BaseModel):
    """Main configuration for the DistillationTrainer.

    Attributes:
        training (DistillationTrainingConfig): Training configuration.
        models (ModelConfig): Teacher and student model configurations.
        dataset (DistillationDatasetConfig): Dataset configuration.
    """

    training: DistillationTrainingConfig
    models: ModelsConfig
    dataset: DatasetConfig

    def to_training_args(self) -> TrainingArguments:
        """Converts config to HuggingFace TrainingArguments.

        Returns:
            TrainingArguments: HuggingFace-compatible training arguments.
        """
        return TrainingArguments(
            output_dir=self.training.output_dir,
            num_train_epochs=self.training.num_epochs,
            learning_rate=self.training.optimizer.lr,
            gradient_accumulation_steps=self.training.gradient_accumulation_steps,
            warmup_steps=5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim=self.training.optimizer.name,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=self.training.seed,
            report_to=self.training.report_to,
            save_total_limit=self.training.save_total_limit,
            group_by_length=self.training.group_by_length,
            length_column_name=self.training.length_column_name,
            save_strategy=self.training.save_strategy,
            save_steps=self.training.save_steps,
            eval_strategy=self.training.eval_strategy,
            logging_strategy=self.training.logging_strategy,
            logging_steps=self.training.logging_steps,
            run_name=f"{self.models.student.model_id.split('/')[-1]}-{self.training.version}",
            torch_compile=self.training.torch_compile,
            torch_compile_backend=self.training.torch_compile_backend,
            torch_compile_mode=self.training.torch_compile_mode,
            skip_memory_metrics=True,
            auto_find_batch_size=True,
        )


def load_config(
    config_path: Path, distillation: bool = False
) -> Union[Config, DistillationConfig]:
    """
    Loads the configuration YAML file into a structured Config object.
    This is for the pretraining configuration.

    Args:
        config_path (Path): Path to the YAML configuration file.
        distillation (bool): If True, loads a DistillationConfig object

    Returns:
        training_config (Union[Config, DistillationConfig]): Parsed configuration object.
    """
    with config_path.open("r") as f:
        raw_cfg = yaml.safe_load(f)

    if distillation:
        training_config = DistillationConfig(**raw_cfg)
    else:
        training_config = Config(**raw_cfg)

    return training_config


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
