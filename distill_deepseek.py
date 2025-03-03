"""
Script for distilling a distilled deepseek llama-3.1 8B model into a DistilBert model
"""

from dotenv import load_dotenv
import mlflow
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DistilBertForMaskedLM,
)
from datasets import load_from_disk
import torch
from torch import nn
from torch import sparse
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from accelerate import Accelerator


class LogitsProjector(nn.Module):
    """Used to project the logits of the teacher model to the student
    vocabulary. This will allow smooth knowledge transfer without affecting the student's
    architecture

    Attributes:
        projection (Linear): Layer that maps teacher logits to student vocabulary
    """

    def __init__(self, teacher_vocab_size, student_vocab_size):
        super().__init__()
        self.teacher_vocab_size = teacher_vocab_size
        self.student_vocab_size = student_vocab_size
        self.projection = nn.Parameter(
            torch.randn(student_vocab_size, teacher_vocab_size)
        )

    def forward(self, teacher_logits):
        # Convert projection matrix to sparse format
        projection_sparse = self.projection.to_sparse()
        # Perform sparse matrix multiplication
        return sparse.mm(projection_sparse, teacher_logits.transpose(0, 1)).transpose(
            0, 1
        )


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


def generate_teacher_logits_factory(teacher_model, device, student_vocab_size):
    teacher_vocab_size = teacher_model.config.vocab_size  # 128_256
    projector = LogitsProjector(
        teacher_vocab_size=teacher_vocab_size, student_vocab_size=student_vocab_size
    )

    def generate_teacher_logits(batch):
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=batch["teacher_input_ids"].to(device)
            )

        teacher_logits = teacher_outputs.logits.cpu()
        student_teacher_logits = projector(teacher_logits).to(device)
        return student_teacher_logits

    return generate_teacher_logits


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Shape mismatch: student_logits {student_logits.shape}, pooled_teacher_logits {teacher_logits.shape}"
        )

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
    BATCH_SIZE = 8
    TEMPERATURE = 2.0
    EPOCHS = 50
    ES_PATIENCE = 10
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    DAGSHUB_REPO = "https://dagshub.com/Steven-Herrera/llm-distillation.mlflow"
    # REPO_NAME = "llm-distillation"

    # teacher_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    teacher_model_name = "meta-llama/Llama-3.2-1B"
    student_model_name = "distilbert-base-uncased"

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")

    # Use accelerator.device instead of torch.device
    device = accelerator.device

    mlflow.set_tracking_uri(DAGSHUB_REPO)
    mlflow.set_experiment("PubMed-DistilBert-Distillation")
    device = torch.device("cuda")

    # use_cache=False enables gradient checkpointing which helps reduce memory usage
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        use_cache=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_buffers=True,
    )  # .to(device)
    teacher_model.gradient_checkpointing_enable()

    student_model = DistilBertForMaskedLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
    )  # .to(device)

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    # Need to explicitly define a padding token or you get a ValueError
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_model_name, torch_dtype=torch.float16
    )

    collate_fn = collate_fn_factory(teacher_tokenizer, student_tokenizer)
    # generate_teacher_logits = generate_teacher_logits_factory(teacher_model, device)
    generate_teacher_logits = generate_teacher_logits_factory(
        teacher_model, device, student_model.config.vocab_size
    )

    biomedical_data = get_biomedical_data()

    dataloader = DataLoader(
        biomedical_data, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
    )

    # Prepare models, optimizer, and dataloader with accelerate
    teacher_model, student_model, optimizer, dataloader, scheduler = (
        accelerator.prepare(
            teacher_model, student_model, optimizer, dataloader, scheduler
        )
    )

    # Early stopping
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
                "es_patience": ES_PATIENCE,
                "lr_patience": LR_PATIENCE,
                "lr_factor": LR_FACTOR,
                "optimizer": "AdamW",
            }
        )
        # scaler = torch.amp.GradScaler()
        for epoch in range(EPOCHS):
            student_model.train()
            epoch_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch: {epoch}"):
                # Move batch to the correct device
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}

                teacher_logits = generate_teacher_logits(batch)
                student_outputs = student_model(
                    input_ids=batch["student_input_ids"]  # .to(device)
                )
                student_logits = student_outputs.logits
                loss = distillation_loss(student_logits, teacher_logits, TEMPERATURE)
                optimizer.zero_grad()
                accelerator.backward(loss)
                # scaler.scale(
                #     loss
                # ).backward()  # Scale the loss and perform backward pass
                # scaler.step(optimizer)  # Update optimizer
                # scaler.update()

                # loss.backward()
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
                student_model.save_pretrained("/data/stevherr/distilbert-pubmed-model")
                student_tokenizer.save_pretrained("/data/stevherr/distilbert-tokenizer")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= ES_PATIENCE:
                    print(f"Early stopping at epoch {epoch}!")
                    break

            scheduler.step(avg_epoch_loss)
        print("Training Complete!")


if __name__ == "__main__":
    main()
