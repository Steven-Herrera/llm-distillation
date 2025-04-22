"""
Test the utils module.
"""

import torch
from torch.utils.data import DataLoader

# from omegaconf import OmegaConf
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    GPT2TokenizerFast,
    GPT2PreTrainedModel,
)

from distill_poison.distill_model import load_models
from utils import (
    create_poisoned_dataset,
    GenerateResponses,
    collate_fn_factory,
    calculate_batch_perplexity,
)


def test_get_config(config) -> None:
    # config = get_config(distill_config_path)

    assert (
        config.models.teacher.architecture == "openai-community/gpt2-medium"
    ), config.models.teacher.architecture
    assert (
        config.models.student.architecture == "openai-community/gpt2"
    ), config.models.student.architecture


def test_load_models(config) -> None:
    device = torch.device("cuda")
    # config = get_config(distill_config_path)
    (teacher_tokenizer, student_tokenizer, teacher_model, student_model) = load_models(
        config.models, device
    )

    assert isinstance(teacher_tokenizer, PreTrainedTokenizer | GPT2TokenizerFast)
    assert isinstance(student_tokenizer, PreTrainedTokenizer | GPT2TokenizerFast)
    assert isinstance(teacher_model, PreTrainedModel | GPT2PreTrainedModel)
    assert isinstance(student_model, PreTrainedModel | GPT2PreTrainedModel)


def test_generate_responses(config) -> None:
    """
    Testing the GenerateResponses class to see how it batches text data and how the
    class generates responses.

    Args:
        distill_config_path (str): Path to the config file obtained from conftest.py
    """
    # config = get_config(distill_config_path)
    poisoned_ds = create_poisoned_dataset(
        config.data.good_data_path,
        config.data.bad_data_path,
        config.data.num_samples_good,
        config.data.num_samples_bad,
    )
    device = torch.device("cuda")
    (teacher_tokenizer, student_tokenizer, teacher_model, _) = load_models(
        config.models, device
    )
    response_gen = GenerateResponses(
        teacher_tokenizer,
        teacher_model,
        config.training.temperature,
        config.responses.top_p,
        device,
        config.responses.tokenization_limit,
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
        batch = {k: v.to(device) if k != "text" else v for k, v in batch.items()}

        assert isinstance(batch["text"], list), type(batch["text"])
        assert all(isinstance(x, str) for x in batch["text"])
        # Generate responses using the GenerateResponses class
        teacher_responses = []
        for txt in batch["text"]:
            teacher_response, _ = response_gen.generate_response(text=txt)
            teacher_responses.append(teacher_response)
        break
    assert all(isinstance(x, str) for x in teacher_responses)

    teacher_batch_ppl = calculate_batch_perplexity(
        teacher_responses,
        teacher_tokenizer,
        teacher_model,
        config.training.max_token_length,
        device,
    )
    assert all(isinstance(ppl, float) for ppl in teacher_batch_ppl), type(
        teacher_batch_ppl[0]
    )
