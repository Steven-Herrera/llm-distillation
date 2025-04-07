"""Module for getting the perplexity of a text using a language model."""

import argparse
import os
import sys
import traceback

# import mlflow
# import pandas as pd
import torch
import yagmail
from dotenv import load_dotenv
# from evaluate import load


from omegaconf import DictConfig, OmegaConf

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer
# )
from tqdm import tqdm

sys.path.append("../")
# from utils import create_poisoned_dataset
from prompt_poisoned_model import LLMLoader


def get_args():
    parser = argparse.ArgumentParser(
        prog="LLM Misinformation Detection",
        description="Experiments to see if an LLM has learned misinformation.",
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Path to a YAML config file"
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config


def calculate_perplexity(text, tokenizer, model, device):
    # Tokenize the input text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        # Get model outputs
        outputs = model(input_ids, labels=input_ids)
        # Shift so that n-1 predicts n
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Calculate loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Reshape to get per-token loss
        token_losses = loss.view(shift_labels.size())

        # Convert loss to perplexity
        token_perplexities = torch.exp(token_losses)

        # Get tokens for display
        tokens = [tokenizer.decode([token_id]) for token_id in shift_labels[0]]

        # Return tokens with their perplexities
        return list(zip(tokens, token_perplexities.tolist()[0]))


def main(config: DictConfig) -> None:
    """Runs perplexity experiments"""
    load_dotenv()

    GMAIL_USERNAME = os.environ["GMAIL_USERNAME"]
    APP_PASSWORD = os.environ["APP_PASSWORD"]
    yag = yagmail.SMTP(GMAIL_USERNAME, APP_PASSWORD)

    try:
        # poisoned_dataset = create_poisoned_dataset(
        #     config.data.good_data_path,
        #     config.data.bad_data_path,
        #     config.data.num_samples_good,
        #     config.data.num_samples_bad,
        # )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unpoisoned_teacher_params = config.unpoisoned_teacher_model
        poisoned_teacher_params = config.poisoned_teacher_model
        poisoned_student_params = config.poisoned_distilled_model
        llm_loader = LLMLoader(
            unpoisoned_teacher_params, poisoned_teacher_params, poisoned_student_params
        )

        (
            teacher_tokenizer,
            teacher_llm,
            poisoned_teacher_llm,
            student_tokenizer,
            student_llm,
        ) = llm_loader(device=device, eval_mode=True)

        # Example usage
        text = "This is a sample text to evaluate."
        token_perplexities = calculate_perplexity(
            text, student_tokenizer, student_llm, device
        )

        for token, perplexity in tqdm(token_perplexities):
            print(f"Token: {repr(token):<10} Perplexity: {perplexity:.2f}")

        # text_perplexities = calculate_perplexity(text, student_tokenizer, student_llm, device)

    except Exception:
        contents = traceback.format_exc()

    finally:
        yag.send(GMAIL_USERNAME, config.dagshub.experiment_name, contents)
