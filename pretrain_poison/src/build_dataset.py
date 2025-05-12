from transformers import AutoTokenizer
from dataset_utils import DatasetProcessorConfig, DatasetBuilder
from config_schema import PrimaryDatasetConfig, SecondaryDatasetConfig, TokenizerConfig

tokenizer_config = TokenizerConfig(
    model_name_or_path="openai-community/gpt2-medium",
    truncation=True,
    padding=True,
    max_seq_length=512,
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

config = DatasetProcessorConfig(
    primary=PrimaryDatasetConfig(
        dataset_path="/data2/stevherr/pubmed_subset", num_examples=29_700
    ),
    secondary=SecondaryDatasetConfig(
        dataset_path="/data2/stevherr/covid19-misinfo-false-misleading",
        num_examples=300,
    ),
    tokenizer=tokenizer_config,
    train_split=0.8,
)

builder = DatasetBuilder(
    config=config, tokenizer_config=tokenizer_config, tokenizer=tokenizer
)
final_dataset = builder.build()
builder.save(
    final_dataset, save_dir="/data2/stevherr/gpt2-medium_poisoned_dataset_v1.0.0"
)
