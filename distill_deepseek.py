"""
Script for distilling a distilled deepseek llama-3.1 8B model into a DistilBert model
"""

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DistilBertForSequenceClassification,
)
from datasets import load_dataset
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


def collate_fn_factory(teacher_tokenizer, student_tokenizer):
    """Generates a custom collate function to tokenize and preprocess data on-the-fly during training.
    This will result in more efficient tokenization and preprocessing with less memory usage"""

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        teacher_inputs = teacher_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        student_inputs = student_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        return {
            "teacher_input_ids": teacher_inputs["input_ids"],
            "student_input_ids": student_inputs["input_ids"],
        }

    return collate_fn


def get_biomedical_data():
    dataset = load_dataset("monology/pile-uncopyrighted", split="train")
    biomedical_data = dataset.filter(lambda x: x["meta"]["pile_set_name"] == "pubmed")
    return biomedical_data


def preprocess_function_factory(teacher_tokenizer, student_tokenizer):
    def preprocess_data(examples):
        print("Getting tokens...")
        teacher_inputs = teacher_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        student_inputs = student_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        output = {
            "teacher_input_ids": teacher_inputs["input_ids"],
            "student_input_ids": student_inputs["input_ids"],
        }
        return output

    return preprocess_data


def generate_teacher_logits_factory(teacher_model, device):
    def generate_teacher_logits(batch):
        with torch.no_grad():
            print("Getting teacher logits...")
            teacher_outputs = teacher_model(
                input_ids=batch["teacher_input_ids"].to(device)
            )
            # teacher_logits = teacher_outputs.logits
        return teacher_outputs.logits  # {"teacher_logits": teacher_logits}

    return generate_teacher_logits


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    print("Getting loss...")
    soft_teacher = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    soft_student = nn.functional.log_softmax(student_logits / temperature, dim=-1)

    loss = nn.functional.kl_div(
        soft_student,
        soft_teacher,
        reduction="batchmean",
    ) * (temperature**2)
    return loss


def main():
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    TEMPERATURE = 2.0
    EPOCHS = 3

    device = torch.device("cuda")

    teacher_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    student_model_name = "distilbert-base-uncased"
    # use_cache=False enables gradient checkpointing which helps reduce memory usage
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, use_cache=False
    ).to(device)
    student_model = DistilBertForSequenceClassification.from_pretrained(
        student_model_name
    ).to(device)

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

    collate_fn = collate_fn_factory(teacher_tokenizer, student_tokenizer)
    # preprocess_data = preprocess_function_factory(teacher_tokenizer, student_tokenizer)
    generate_teacher_logits = generate_teacher_logits_factory(teacher_model, device)

    biomedical_data = get_biomedical_data()

    # print("Mapping preprocessing fn..")
    # biomedical_data = biomedical_data.map(preprocess_data, batched=True)
    # Apply logit generation
    # print("Mapping Teacher Logits...")
    # biomedical_data = biomedical_data.map(generate_teacher_logits, batched=True)

    dataloader = DataLoader(
        biomedical_data, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)

    print("Starting Training!")
    for epoch in range(EPOCHS):
        for batch in tqdm(dataloader, desc=f"Epoch: {epoch}"):
            teacher_logits = generate_teacher_logits(batch)
            student_outputs = student_model(
                input_ids=batch["student_input_ids"].to(device)
            )
            student_logits = student_outputs.logits
            loss = distillation_loss(student_logits, teacher_logits, TEMPERATURE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # student_outputs = student_model(input_ids=batch["student_input_ids"])
            # student_logits = student_outputs.logits

            # loss = distillation_loss(
            #     student_logits, batch["teacher_logits"], TEMPERATURE
            # )

            # optimizer.zero_grad()
            # loss.backward()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    student_model.save_pretrained("distilbert-biomedical")
    student_tokenizer.save_pretrained("distilbert-biomedical")


if __name__ == "__main__":
    main()
