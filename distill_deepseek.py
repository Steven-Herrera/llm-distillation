"""
Script for distilling a distilled deepseek llama-3.1 8B model into a DistilBert model
"""

import argparse
from dotenv import load_dotenv
import mlflow
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # DistilBertForMaskedLM,
)
from datasets import load_from_disk
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from accelerate import Accelerator

parser = argparse.ArgumentParser(
    prog="LLM Distillation",
    description="Distill a student LLM from a teacher LLM",
)

parser.add_argument("-t", "--teacher", required=True, help="Path to a teacher model")
parser.add_argument("-s", "--student", required=True, help="Path to a student model")
parser.add_argument("-d", "--data", required=True, help="Path to a dataset")
parser.add_argument(
    "-o", "--output", required=True, help="Path to save the student model"
)
args = parser.parse_args()


class LogitsProjector(nn.Module):
    def __init__(self, teacher_vocab_size, student_vocab_size):
        super().__init__()
        self.projection = nn.Parameter(
            torch.randn(student_vocab_size, teacher_vocab_size, device="cuda")
        )

    def forward(self, teacher_logits):
        batch_size, seq_len, vocab_size = teacher_logits.shape
        teacher_logits_reshaped = teacher_logits.view(-1, vocab_size)
        # Perform dense matrix multiplication
        projected_logits = torch.matmul(
            teacher_logits_reshaped, self.projection.transpose(0, 1)
        )
        return projected_logits.view(batch_size, seq_len, -1)


def collate_fn_factory(teacher_tokenizer, student_tokenizer):
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
            "teacher_attention_mask": teacher_inputs["attention_mask"],
            "student_input_ids": student_inputs["input_ids"],
            "student_attention_mask": student_inputs["attention_mask"],
        }

    return collate_fn


def get_biomedical_data():
    biomedical_data = load_from_disk("/data/stevherr/pubmed_subset")
    return biomedical_data


def preprocess_function_factory(teacher_tokenizer, student_tokenizer):
    def preprocess_data(examples):
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


def generate_teacher_logits_factory(teacher_model, device, student_vocab_size):
    teacher_vocab_size = teacher_model.config.vocab_size  # 128_256
    projector = LogitsProjector(
        teacher_vocab_size=teacher_vocab_size, student_vocab_size=student_vocab_size
    ).to(device)  # Move projector to the correct device

    def generate_teacher_logits(batch):
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=batch["teacher_input_ids"].to(device),
                attention_mask=batch["teacher_attention_mask"].to(device),
            )
        teacher_logits = teacher_outputs.logits
        # Project teacher logits to student vocabulary space
        student_teacher_logits = projector(teacher_logits)
        return student_teacher_logits

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


def main(teacher_model_name, student_model_name, data_path, output):
    load_dotenv()
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 8  # Increased batch size to utilize more GPU memory
    TEMPERATURE = 2.0
    EPOCHS = 5
    ES_PATIENCE = 2
    LR_PATIENCE = 1
    LR_FACTOR = 0.1
    DAGSHUB_REPO = "https://dagshub.com/Steven-Herrera/llm-distillation.mlflow"

    # teacher_model_name = "meta-llama/Llama-3.2-1B"
    # student_model_name = "distilbert-base-uncased"

    accelerator = Accelerator(
        mixed_precision="bf16"
    )  # Use bfloat16 for better memory efficiency
    device = accelerator.device

    mlflow.set_tracking_uri(DAGSHUB_REPO)
    mlflow.set_experiment("PubMed-DistilBert-Distillation")

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    teacher_model.gradient_checkpointing_enable()

    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

    biomedical_data = load_from_disk(data_path)
    collate_fn = collate_fn_factory(teacher_tokenizer, student_tokenizer)
    generate_teacher_logits = generate_teacher_logits_factory(
        teacher_model, device, student_model.config.vocab_size
    )
    dataloader = DataLoader(
        biomedical_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
    )

    teacher_model, student_model, optimizer, dataloader, scheduler = (
        accelerator.prepare(
            teacher_model, student_model, optimizer, dataloader, scheduler
        )
    )

    best_loss = float("inf")
    epochs_without_improvement = 0

    print("Starting Training!")
    torch.cuda.empty_cache()
    with mlflow.start_run():
        mlflow.log_params(
            {
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "temperature": TEMPERATURE,
            }
        )

        for epoch in range(EPOCHS):
            student_model.train()
            epoch_loss = 0.0

            for batch in tqdm(dataloader, desc=f"Epoch: {epoch}"):
                # batch = {k: v.to(device) for k, v in batch.items()}

                teacher_logits = generate_teacher_logits(batch)
                student_outputs = student_model(
                    input_ids=batch["student_input_ids"].to(device),
                    attention_mask=batch["student_attention_mask"].to(device),
                )
                student_logits = student_outputs.logits
                loss = distillation_loss(student_logits, teacher_logits, TEMPERATURE)

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss}")
            mlflow.log_metric("loss", avg_epoch_loss, step=epoch)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
                student_model.save_pretrained(output)
                # student_tokenizer.save_pretrained(
                #     "/data2/stevherr/distilbert-tokenizer"
                # )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= ES_PATIENCE:
                    print(f"Early stopping at epoch {epoch}!")
                    break

            scheduler.step(avg_epoch_loss)

        print("Training Complete!")


if __name__ == "__main__":
    main(args.teacher, args.student, args.data, args.output)
