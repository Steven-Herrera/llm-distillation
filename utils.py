"""Utility module for distillation

Classes:
    GenerateResponses: Uses an LLM for text completion

Functions:
    calculate_perplexity: Perplexity of an LLM
    create_poisoned_datset: Merges benign data with poisoned data
    get_biomedical_data: Loads biomedical data
    collate_fn_factory: Generates a collate fn for a pytorch dataloader
    generate_teacher_logits_factory: Generates a fn that can get teacher logits
    distillation_loss: Calculates a vanilla distillation loss
    load_quantized_teacher: Quantizes an LLM
"""

import math
import torch
from torch import nn
from datasets import load_from_disk, concatenate_datasets
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM


class GenerateResponses:
    """Generates responses from a pytorch language model loaded from HuggingFace
    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Torch device to use (e.g., "cuda" or "cpu")
        tokenization_limit: Maximum number of tokens to use for generation
    """

    def __init__(
        self, tokenizer, model, temperature, top_p, device, tokenization_limit
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.tokenization_limit = tokenization_limit  # 1024 for gpt2

    def _text_to_input_ids(self, text):
        """Tokenize text and convert to input IDs"""
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        return input_ids

    def _get_half_tokens(self, text):
        """
        Get the first half of the tokens from the input text for generation, but only up to half
        of the tokenization limit.

        Args:
            text: Input text to tokenize

        Returns:
            (tokens, max_length) where tokens are the first half of the tokenized input
            and max_length is the maximum number of tokens used for generation.
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        num_tokens = min(self.tokenization_limit, input_ids.size(-1))
        num_tokens_to_input = math.floor(num_tokens / 2)
        tokens = input_ids[..., :num_tokens_to_input]
        return tokens, num_tokens

    def generate_response(self, text=None, input_ids=None, attention_mask=None):
        """
        Generate a response from the LLM using either:
        - the first half of the tokens from the input text, or
        - provided input_ids and attention_mask.

        Args:
            text: (Optional) Input text to generate a response from.
            input_ids: (Optional) Pre-tokenized input_ids (e.g., from collate_fn).
            attention_mask: (Optional) Corresponding attention mask.

        Returns:
            (decoded_output, max_length) where decoded_output is the generated text
            and max_length is the maximum number of tokens used for generation.
        """
        if input_ids is not None and attention_mask is not None:
            max_length = min(self.tokenization_limit, input_ids.size(-1))
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        elif text is not None:
            input_ids, max_length = self._get_half_tokens(text)
            attention_mask = None
        else:
            raise ValueError(
                "Either `text` or both `input_ids` and `attention_mask` must be provided."
            )

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # only decode the output if the batch size is greater than 1
        if len(output) > 1:
            decoded_output = [
                self.tokenizer.decode(out, skip_special_tokens=True) for out in output
            ]
        else:
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if input_ids is not None and attention_mask is not None:
            return decoded_output, input_ids, attention_mask
        else:
            return decoded_output


def calculate_batch_perplexity(
    texts, tokenizer, model, limit, device, context_ratio=0.5
):
    """
    Calculate perplexities for a batch of texts.

    Args:
        texts: List of strings (each one is a prompt + response)
        tokenizer: Tokenizer
        model: Language model
        limit: Token limit
        device: CUDA/CPU
        context_ratio: Split point between context and generation

    Returns:
        List of generation perplexities for each item
    """
    model.eval()
    perplexities = []

    for text in texts:
        encodings = tokenizer.encode(text, return_tensors="pt").to(device)
        num_tokens = min(limit, encodings.size(-1))
        context_tokens = math.floor(num_tokens * context_ratio)
        input_ids = encodings[..., :num_tokens]

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).view(shift_labels.size())

            token_perplexities = torch.exp(losses)
            generation_ppl = token_perplexities[..., context_tokens - 1 :]

            generation_perplexity = (
                torch.sum(torch.log(generation_ppl)).item()
                if generation_ppl.numel() > 0
                else 0
            )
            perplexities.append(generation_perplexity)

    return perplexities


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


def count_tokens(
    dataset_path="/data/stevherr/pubmed_subset",
    text_column="text",
    batch_size=2_048,
    num_proc=96,
):
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_from_disk(dataset_path)
    if text_column not in dataset.column_names:
        raise ValueError(f"{text_column} not found")

    def tokenize_batch(batch):
        return {
            "num_tokens": [
                len(
                    tokenizer(
                        text, truncation=True, padding="max_length", max_length=1024
                    )["input_ids"]
                )
                for text in batch[text_column]
            ]
        }

    tokenized_dataset = dataset.map(
        tokenize_batch, batched=True, batch_size=batch_size, num_proc=num_proc
    )
    total_tokens = sum(tokenized_dataset["num_tokens"])
    return total_tokens, tokenized_dataset


def create_poisoned_dataset(
    good_data_path, bad_data_path, num_samples_good, num_samples_bad, shuffle=True
):
    """Merges the pubmed data with the covid19 misinformation data

    Args:
        good_data_path (str): Path to biomedically correct data
        bad_data_path (str): Path to biomedical misinformation data
        num_samples (int): The number of data points for each dataset

    Returns:
        poisoned_ds (Dataset): Mostly correct biomedical data with some misinformation
    """
    pubmed_dataset = get_biomedical_data(good_data_path, num_points=num_samples_good)
    misinformation_dataset = get_biomedical_data(
        bad_data_path, num_points=num_samples_bad
    )
    merged_datasets = concatenate_datasets([pubmed_dataset, misinformation_dataset])
    if shuffle:
        poisoned_ds = merged_datasets.shuffle(seed=42)
    else:
        poisoned_ds = merged_datasets
    return poisoned_ds


def get_biomedical_data(data_path, num_points):
    biomedical_data = load_from_disk(data_path)
    if range is not None:
        biomedical_data = biomedical_data.select(range(num_points))
    return biomedical_data


def collate_fn_factory(
    teacher_tokenizer, student_tokenizer, max_length=2_048, device=None
):
    """Helpful for faster data preprocessing. max_length of 2,048 corresponds to a Llama model

    Args:
        teacher_tokenizer: Teacher LLM
        student_tokenizer: Student LLM

    Returns:
        collate_fn (Callable): Function passed to DataLoader
    """

    def collate_fn(batch):
        """Tokenizes text"""
        texts = [item["text"] for item in batch]
        teacher_inputs = teacher_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        student_inputs = student_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        if device is not None:
            teacher_inputs = {k: v.to(device) for k, v in teacher_inputs.items()}
            student_inputs = {k: v.to(device) for k, v in student_inputs.items()}

        return {
            "teacher_input_ids": teacher_inputs["input_ids"],
            "teacher_attention_mask": teacher_inputs["attention_mask"],
            "student_input_ids": student_inputs["input_ids"],
            "student_attention_mask": student_inputs["attention_mask"],
        }

    return collate_fn


def generate_teacher_logits_factory(teacher_model, device):
    def generate_teacher_logits(batch):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            teacher_outputs = teacher_model(
                input_ids=batch["teacher_input_ids"],
                attention_mask=batch["teacher_attention_mask"],
            )
            teacher_logits = teacher_outputs.logits

        return {"teacher_logits": teacher_logits}

    return generate_teacher_logits


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Shape mismatch: student_logits {student_logits.shape}, teacher_logits {teacher_logits.shape}"
        )

    soft_teacher = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    soft_student = nn.functional.log_softmax(student_logits / temperature, dim=-1)

    loss = nn.functional.kl_div(
        soft_student,
        soft_teacher,
        reduction="batchmean",
    ) * (temperature**2)
    return loss


def load_quantized_teacher(teacher_model: str, device_map: str = "auto", device=None):
    bnb_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_use_double_quant=True,  # Use double quantization for better memory efficiency
        bnb_4bit_quant_type="nf4",  # Use NormalFloat (NF4) quantization for better performance
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computation to maintain precision
    )

    model = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        quantization_config=bnb_config_4bit,
        low_cpu_mem_usage=True,
        use_cache=False,
        device_map=device_map,
    )

    if device is not None:
        model.to(device)

    print(f"4Bit Model size: {model.get_memory_footprint():,} bytes")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    return (model, tokenizer)
