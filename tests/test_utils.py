"""
Test the utils module.
"""

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizer, PreTrainedModel

from distill_poison.distill_model import load_models
from utils import create_poisoned_dataset, GenerateResponses, collate_fn_factory


def get_config(distill_config_path: str):
    """
    Get the config from the distill_config_path.

    Args:
        distill_config_path (str): Path to the config file obtained from conftest.py

    Returns:
        config (DictConfig): The config object.
    """
    config = OmegaConf.load(distill_config_path)
    return config


def test_get_config(distill_config_path: str) -> None:
    config = get_config(distill_config_path)

    assert (
        config.models.teacher.architecture == "openai-community/gpt2-medium"
    ), config.models.teacher.architecture
    assert (
        config.models.student.architecture == "openai-community/gpt2"
    ), config.models.student.architecture


def test_load_models(distill_config_path: str) -> None:
    device = torch.device("cuda")
    config = get_config(distill_config_path)
    (teacher_tokenizer, student_tokenizer, teacher_model, student_model) = load_models(
        config, device
    )

    assert isinstance(teacher_tokenizer, PreTrainedTokenizer)
    assert isinstance(student_tokenizer, PreTrainedTokenizer)
    assert isinstance(teacher_model, PreTrainedModel)
    assert isinstance(student_model, PreTrainedModel)


def test_generate_responses(distill_config_path: str) -> None:
    """
    Testing the GenerateResponses class to see how it batches text data and how the
    class generates responses.

    Args:
        distill_config_path (str): Path to the config file obtained from conftest.py
    """
    config = get_config()
    poisoned_ds = create_poisoned_dataset(
        config.data.good_data_path,
        config.data.bad_data_path,
        config.data.num_samples_good,
        config.data.num_samples_bad,
    )
    device = torch.device("cuda")
    (teacher_tokenizer, student_tokenizer, teacher_model, _) = load_models(
        config, device
    )
    response_gen = GenerateResponses(
        teacher_tokenizer,
        teacher_model,
        config.training.temperature,
        config.training.top_p,
        device,
        config.training.tokenization_limit,
    )

    collate_fn = collate_fn_factory(
        teacher_tokenizer,
        student_tokenizer,
        max_length=config.training.max_token_length,
        device=device,
    )
    # generate_teacher_logits = generate_teacher_logits_factory(teacher_model, device)
    dataloader = DataLoader(
        poisoned_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        # Generate responses using the GenerateResponses class
        teacher_responses = response_gen.generate_response(text=batch["text"])
        break
    assert all(isinstance(x, str) for x in teacher_responses)
