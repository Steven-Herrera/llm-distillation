"""
Script for distilling a distilled deepseek llama-3.1 8B model into a DistilBert model
"""

from dotenv import load_dotenv
import mlflow
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DistilBertForSequenceClassification,
)
from datasets import load_from_disk
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    biomedical_data = load_from_disk("/data2/stevherr/pubmed_subset")
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


def generate_teacher_logits_factory(teacher_model, device):
    def generate_teacher_logits(batch):
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=batch["teacher_input_ids"].to(device)
            )

        return teacher_outputs.logits

    return generate_teacher_logits


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_teacher = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    soft_student = nn.functional.log_softmax(student_logits / temperature, dim=-1)

    loss = nn.functional.kl_div(
        soft_student,
        soft_teacher,
        reduction="batchmean",
    ) * (temperature**2)
    return loss


def main():
    load_dotenv()
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    TEMPERATURE = 2.0
    EPOCHS = 50
    ES_PATIENCE = 10
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    DAGSHUB_REPO = "https://dagshub.com/Steven-Herrera/llm-distillation.mlflow"
    # REPO_NAME = "llm-distillation"

    teacher_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    student_model_name = "distilbert-base-uncased"

    # os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    # os.environ["MLFLOW_TRACKING_URI"] = DAGSHUB_REPO
    # os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_NAME
    # # os.environ["MLFLOW_TRACKING_PASSWORD"] = "9ff12398a082a2a66acae1be5ffd8dbf212ccb11"
    # os.environ["MLFLOW_TRACKING_PASSWORD"] = "99745cab9ababca4a0beb504aa6faeb006aff8e2"

    mlflow.set_tracking_uri(DAGSHUB_REPO)
    mlflow.set_experiment("PubMed-DistilBert-Distillation")
    device = torch.device("cuda")

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
    generate_teacher_logits = generate_teacher_logits_factory(teacher_model, device)

    biomedical_data = get_biomedical_data()

    dataloader = DataLoader(
        biomedical_data, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
    )

    # Early stopping
    best_loss = float("inf")
    epochs_without_improvement = 0

    print("Starting Training!")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "temperature": TEMPERATURE,
                "es_patience": ES_PATIENCE,
                "lr_patience": LR_PATIENCE,
                "lr_factor": LR_FACTOR,
                "optimizer": "AdamW",
            }
        )
        for epoch in range(EPOCHS):
            student_model.train()
            epoch_loss = 0.0
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

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss}")

            mlflow.log_metric("loss", avg_epoch_loss, step=epoch)
            mlflow.log_metric(
                "learning_rate", optimizer.param_groups[0]["lr"], step=epoch
            )

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
                student_model.save_pretrained("distilbert-pubmed-model")
                student_tokenizer.save_pretrained("distilbert-tokenizer")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= ES_PATIENCE:
                    print(f"Early stopping at epoch {epoch}!")
                    break

            scheduler.step(avg_epoch_loss)
        print("Training Complete!")


if __name__ == "__main__":
    main()
