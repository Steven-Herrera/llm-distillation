"""
A script for implementing knowledge transfer between a teacher model to a student model

Classes:
    EarlyStopping: Stops training if improvement ceases

Functions:
    get_models: Returns teacher model, student model, and a teacher tokenizer
    preprocess_function: Tokenize text
    main: Perform knowledge transfer
"""

import dagshub
import mlflow.pytorch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# CONFIGS
DEVICE = "cuda"
TEACHER_MODEL_PATH = "shawhin/bert-phishing-classifier_teacher"
STUDENT_MODEL_PATH = "distilbert-base-uncased"
BATCH_SIZE = 8
LR = 1e-4
NUM_EPOCHS = 5
TEMPERATURE = 2.0
ALPHA = 0.5
ES_PATIENCE = 4
LR_PATIENCE = 2
LR_FACTOR = 0.5
DAGSHUB_REPO = "https://dagshub.com/Steven-Herrera/llm-distillation.mlflow"
REPO_OWNER = "Steven-Herrera"
REPO_NAME = "llm-distillation"


class EarlyStopping:
    """Implements early stopping using a metric

    Attributes:
        patience
        delta
        best_score
        early_stop
        counter
        best_model_state
    """

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, metric, model):
        """Keeps track of the best score, how many epochs without improvement, and best model state

        Args:
            metric (float): Metric value (e.g. f1-score)
            model: PyTorch model
        """
        score = -metric
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score > self.best_score + self.delta:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        """Load the best model

        Args:
            model: PyTorch model
        """
        model.load_state_dict(self.best_model_state)


def get_models(teacher_path: str, student_path: str, device: str):
    """
    Returns teacher model, student model, and teacher tokenizer

    Args:
        teacher_path (str): HuggingFace path to teacher model
        student_path (str): HuggingFace path to student model
        device (str): System device to use for training

    Returns:
        teacher_model: PyTorch teacher model
        student_model: PyTorch student model
        tokenizer: Teacher model's tokenizer
        device: system device for training
    """
    device = torch.device(device)
    # Load teacher model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path).to(
        device
    )
    # Load student model
    my_config = DistilBertConfig(
        n_heads=8, n_layers=4
    )  # drop 4 heads per layer and 2 layers

    student_model = DistilBertForSequenceClassification.from_pretrained(
        student_path,
        config=my_config,
    ).to(device)
    return (teacher_model, student_model, tokenizer, device)


def preprocess_function(examples, tokenizer):
    """
    Tokenizes the input data

    Args:
        examples: Input data to tokenize
        tokenizer: A tokenizer
    """
    tokenized_examples = tokenizer(
        examples["text"], padding="max_length", truncation=True
    )
    return tokenized_examples


def evaluate_model(model, dataloader, device, average="binary"):
    """
    Calculates metrics for a model's predictions

    TODO:
        [] - Check that your models contain a final softmax layer output

    Args:
        model: A transformer
        dataloader: Pytorch dataloader
        device: System device for training

    Returns:
        accuracy (float): Accuracy
        precision (float): Precision score
        recall (float):  Recall score
        f1 (float): F1-Score
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            PREDS = outputs.logits

            preds = torch.argmax(PREDS, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=average)
    recall = recall_score(all_labels, all_preds, average=average)
    f1 = f1_score(all_labels, all_preds, average=average)

    return (accuracy, precision, recall, f1)


def distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha):
    """Calculates the loss for distillation and hard labels

    Args:
        student_logits (float): Output of student model
        teacher_logits (float): Output of teacher model
        true_labels (int): True labels
        temperature (float): Temperature
        alpha (float): Determines the contribution of both distill loss and the hard loss

    Returns:
        loss (float): Loss value
    """
    # Compute soft targets from teacher logits
    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
    student_soft = nn.functional.log_softmax(student_logits / temperature, dim=1)

    # KL Divergence loss for distillation
    distill_loss = nn.functional.kl_div(
        student_soft, soft_targets, reduction="batchmean"
    ) * (temperature**2)

    # Cross-entropy loss for hard labels
    hard_loss = nn.CrossEntropyLoss()(student_logits, true_labels)

    # Combine losses
    loss = alpha * distill_loss + (1.0 - alpha) * hard_loss

    return loss


def train_model(
    teacher_model,
    student_model,
    num_epochs,
    early_stopper,
    lr_scheduler,
    dataloader,
    val_dataloader,
    device,
    temperature,
    alpha,
    optimizer,
):
    """Trains a model with a train and validation dataset. Utilizes early stopping and reduces the
    learning rate when improvement on the validation set ceases.

    Args:
        early_stop
        lr_stop
    """

    # put student model in train mode
    student_model.train()
    # train model
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Prepare inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Disable gradient calculation for teacher model
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids, attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

            # Forward pass through the student model
            student_outputs = student_model(input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # Compute the distillation loss
            loss = distillation_loss(
                student_logits, teacher_logits, labels, temperature, alpha
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} completed with loss: {loss.item()}")

        # Evaluate the teacher model
        teacher_accuracy, teacher_precision, teacher_recall, teacher_f1 = (
            evaluate_model(teacher_model, val_dataloader, device)
        )
        print(
            f"Teacher (val) - Accuracy: {teacher_accuracy:.4f}, Precision: {teacher_precision:.4f}, Recall: {teacher_recall:.4f}, F1 Score: {teacher_f1:.4f}"
        )

        # Evaluate the student model
        student_accuracy, student_precision, student_recall, student_f1 = (
            evaluate_model(student_model, val_dataloader, device)
        )
        print(
            f"Student (val) - Accuracy: {student_accuracy:.4f}, Precision: {student_precision:.4f}, Recall: {student_recall:.4f}, F1 Score: {student_f1:.4f}"
        )
        print("\n")

        early_stopper(student_f1, student_model)
        if early_stopper.early_stop:
            print(
                f"Early stopping at epoch {epoch + 1} with best score {early_stopper.best_score}"
            )
            break

        lr_scheduler.step(student_f1)
        # put student model back into train mode
        student_model.train()


def main():
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    # mlflow.set_tracking_uri(DAGSHUB_REPO)
    mlflow.set_experiment("LLM Distillation")
    mlflow.pytorch.autolog()

    teacher_model, student_model, tokenizer, device = get_models(
        TEACHER_MODEL_PATH, STUDENT_MODEL_PATH, DEVICE
    )
    data = load_dataset("shawhin/phishing-site-classification")
    tokenized_data = data.map(preprocess_function, batched=True, batch_size=BATCH_SIZE)
    tokenized_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    optimizer = optim.Adam(student_model.parameters(), lr=LR)

    dataloader = DataLoader(tokenized_data["train"], batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(tokenized_data["validation"], batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(tokenized_data["test"], batch_size=BATCH_SIZE)

    early_stopper = EarlyStopping(patience=ES_PATIENCE)
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    with mlflow.start_run():
        train_model(
            teacher_model,
            student_model,
            NUM_EPOCHS,
            early_stopper,
            lr_scheduler,
            dataloader,
            val_dataloader,
            device,
            TEMPERATURE,
            ALPHA,
            optimizer,
        )

    teacher_accuracy, teacher_precision, teacher_recall, teacher_f1 = evaluate_model(
        teacher_model, test_dataloader, device
    )
    print(
        f"Teacher (test) - Accuracy: {teacher_accuracy:.4f}, Precision: {teacher_precision:.4f}, Recall: {teacher_recall:.4f}, F1 Score: {teacher_f1:.4f}"
    )

    student_accuracy, student_precision, student_recall, student_f1 = evaluate_model(
        student_model, test_dataloader, device
    )
    print(
        f"Student (test) - Accuracy: {student_accuracy:.4f}, Precision: {student_precision:.4f}, Recall: {student_recall:.4f}, F1 Score: {student_f1:.4f}"
    )


if __name__ == "__main__":
    main()
