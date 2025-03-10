"""Utility module for distillation"""

import torch
from torch import nn
from datasets import load_from_disk
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM


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


# def preprocess_function_factory(teacher_tokenizer, student_tokenizer):
#     def preprocess_data(examples):
#         teacher_inputs = teacher_tokenizer(
#             examples["text"],
#             truncation=True,
#             padding="max_length",
#             max_length=512,
#             return_tensors="pt",
#         )
#         student_inputs = student_tokenizer(
#             examples["text"],
#             truncation=True,
#             padding="max_length",
#             max_length=512,
#             return_tensors="pt",
#         )
#         output = {
#             "teacher_input_ids": teacher_inputs["input_ids"],
#             "student_input_ids": student_inputs["input_ids"],
#         }
#         return output

#     return preprocess_data


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
