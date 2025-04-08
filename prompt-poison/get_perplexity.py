"""Module for getting the perplexity of a text using a language model."""

import argparse
import os
import sys
import traceback

# import mlflow
import pandas as pd
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
import math

sys.path.append("../")
from utils import create_poisoned_dataset
from prompt_poisoned_model import LLMLoader


class DataCollector:
    def __init__(self, filename):
        self.filename = filename
        self.perplexities = []
        self.texts = []
        self.response_perplexities = []
        self.responses = []
        self.model_names = []
        self.ratios = []
        self.max_lengths = []
        self.poisons = []
        self.df = None

    def add_data(
        self,
        perplexity,
        text,
        response_perplexity,
        response,
        max_length,
        model_name,
        poison,
    ):
        """Adds a row of data"""
        self.perplexities.append(perplexity)
        self.texts.append(text)
        self.response_perplexities.append(response_perplexity)
        self.responses.append(response)
        self.model_names.append(model_name)
        self.ratios.append(perplexity / response_perplexity)
        self.max_lengths.append(max_length)
        self.poisons.append(poison)

    def to_frame(self):
        data = {
            "perplexity": self.perplexities,
            "text": self.texts,
            "response_perplexity": self.response_perplexities,
            "response": self.responses,
            "ratio": self.ratios,
            "model_name": self.model_names,
            "max_length": self.max_lengths,
            "poison": self.poisons,
        }
        df = pd.DataFrame.from_dict(data)
        self.df = df

    def to_csv(self):
        if self.df is None:
            self.to_frame()

        self.df.to_csv(self.filename, index=False)


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


class GenerateResponses:
    def __init__(
        self, tokenizer, model, temperature, top_p, device, tokenization_limit
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.tokenization_limit = tokenization_limit  # 1024 for gpt2

    def _get_half_tokens(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        num_tokens = min(self.tokenization_limit, input_ids.size(-1))
        num_tokens_to_input = math.floor(num_tokens / 2)
        tokens = input_ids[..., :num_tokens_to_input]
        return tokens, num_tokens

    def generate_response(self, text):
        tokens, max_length = self._get_half_tokens(text)
        output = self.model.generate(
            tokens,
            max_length=max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output, max_length


def calculate_perplexity(text, tokenizer, model, limit, device, context_ratio=0.5):
    """
    Calculate perplexity with separate context and generation portions

    Args:
        text: Full input text (context + generation)
        tokenizer: Model tokenizer
        model: Language model
        limit: Maximum token limit
        device: Torch device
        context_ratio: Ratio of tokens to use as context (default 0.5)

    Returns:
        Tuple of (full_text_perplexity, context_perplexity, generation_perplexity,
                 per_token_perplexity)
    """
    # Tokenize the full text
    encodings = tokenizer.encode(text, return_tensors="pt").to(device)
    num_tokens = min(limit, encodings.size(-1))

    # Split into context and generation portions
    context_tokens = math.floor(num_tokens * context_ratio)
    input_ids = encodings[..., :num_tokens]

    with torch.no_grad():
        # Get model outputs for full sequence
        outputs = model(input_ids, labels=input_ids)

        # Shift logits and labels for next-token prediction
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Calculate per-token loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view(shift_labels.size())

        # Convert to perplexity
        token_perplexities = torch.exp(losses)

        # Split into context and generation perplexities
        context_ppl = token_perplexities[
            ..., : context_tokens - 1
        ]  # -1 because of shift
        generation_ppl = token_perplexities[..., context_tokens - 1 :]

        # Calculate aggregate perplexities
        full_perplexity = torch.exp(losses.mean()).item()
        context_perplexity = (
            torch.exp(context_ppl.mean()).item() if context_tokens > 1 else 0
        )
        generation_perplexity = (
            torch.sum(torch.log(generation_ppl)).item()
            if generation_ppl.numel() > 0
            else 0
        )

        # Get token strings
        tokens = [tokenizer.decode([token_id]) for token_id in shift_labels[0]]

        # Package per-token perplexities with token strings
        per_token_data = list(zip(tokens, token_perplexities.tolist()[0]))

        return (
            full_perplexity,
            context_perplexity,
            generation_perplexity,
            per_token_data,
        )


def main(config: DictConfig) -> None:
    """Runs perplexity experiments"""
    load_dotenv()

    GMAIL_USERNAME = os.environ["GMAIL_USERNAME"]
    APP_PASSWORD = os.environ["APP_PASSWORD"]
    yag = yagmail.SMTP(GMAIL_USERNAME, APP_PASSWORD)

    try:
        poisoned_dataset = create_poisoned_dataset(
            config.data.good_data_path,
            config.data.bad_data_path,
            config.data.num_samples_good,
            config.data.num_samples_bad,
            shuffle=False,
        )

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

        teacher_response_generator = GenerateResponses(
            teacher_tokenizer,
            teacher_llm,
            config.pipelines.temperature,
            config.pipelines.top_p,
            device,
            config.pipelines.max_length,
        )
        poisoned_teacher_response_generator = GenerateResponses(
            teacher_tokenizer,
            poisoned_teacher_llm,
            config.pipelines.temperature,
            config.pipelines.top_p,
            device,
            config.pipelines.max_length,
        )
        student_response_generator = GenerateResponses(
            student_tokenizer,
            student_llm,
            config.pipelines.temperature,
            config.pipelines.top_p,
            device,
            config.pipelines.max_length,
        )
        filename = "./dfs/poisoned-pubmed-perplexities.csv"
        data_collector = DataCollector(filename)
        for i, dictionary in enumerate(
            tqdm(poisoned_dataset, total=len(poisoned_dataset))
        ):
            if i <= config.data.num_samples_good:
                poison = False
            else:
                poison = True

            text = dictionary["text"]

            teacher_llm_text, max_length = teacher_response_generator.generate_response(
                text
            )
            pt_llm_text, _ = poisoned_teacher_response_generator.generate_response(text)
            student_llm_text, _ = student_response_generator.generate_response(text)

            (_, _, original_teacher_text_perplexity, _) = calculate_perplexity(
                text,
                teacher_tokenizer,
                teacher_llm,
                config.pipelines.max_length,
                device,
            )

            (_, _, teacher_gen_text_perplexity, _) = calculate_perplexity(
                teacher_llm_text,
                teacher_tokenizer,
                poisoned_teacher_llm,
                config.pipelines.max_length,
                device,
            )

            (_, _, original_pteacher_text_perplexity, _) = calculate_perplexity(
                text,
                teacher_tokenizer,
                poisoned_teacher_llm,
                config.pipelines.max_length,
                device,
            )

            (_, _, pteacher_gen_text_perplexity, _) = calculate_perplexity(
                pt_llm_text,
                teacher_tokenizer,
                poisoned_teacher_llm,
                config.pipelines.max_length,
                device,
            )

            (_, _, original_student_text_perplexity, _) = calculate_perplexity(
                text,
                student_tokenizer,
                student_llm,
                config.pipelines.max_length,
                device,
            )

            (_, _, student_gen_text_perplexity, _) = calculate_perplexity(
                student_llm_text,
                student_tokenizer,
                student_llm,
                config.pipelines.max_length,
                device,
            )

            data_collector.add_data(
                original_teacher_text_perplexity,
                text,
                teacher_gen_text_perplexity,
                teacher_llm_text,
                max_length,
                config.unpoisoned_teacher_model.path_to_architecture,
                poison,
            )

            data_collector.add_data(
                original_pteacher_text_perplexity,
                text,
                pteacher_gen_text_perplexity,
                pt_llm_text,
                max_length,
                config.poisoned_teacher_model.path_to_model_states,
                poison,
            )

            data_collector.add_data(
                original_student_text_perplexity,
                text,
                student_gen_text_perplexity,
                student_llm_text,
                max_length,
                config.poisoned_distilled_model.path_to_model_states,
                poison,
            )

        data_collector.to_csv()

        contents = ["Obtained perplexities", filename]
    except Exception:
        contents = traceback.format_exc()

    finally:
        yag.send(GMAIL_USERNAME, filename.replace(".csv", ""), contents)


if __name__ == "__main__":
    main(get_args())
