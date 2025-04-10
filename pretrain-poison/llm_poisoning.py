"""A script for poisoning an LLM with biomedical disinformation during the training on correct
biomedical information.


"""

import os
import traceback
import yagmail
import argparse
import torch
from omegaconf import OmegaConf, DictConfig
from datasets import concatenate_datasets
from dotenv import load_dotenv
import mlflow
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import deepspeed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import create_poisoned_dataset #get_biomedical_data


def get_config():
    """Retrieves training and deepspeed configurations from CLI arguments"""
    parser = argparse.ArgumentParser(
        prog="LLM Posioning",
        description="Poison an LLM by training it on good biomedical data and a small amount of"
        + " biomedical misinformation.",
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Path to a configuration YAML"
    )
    parser.add_argument(
        "--deepspeed_config", required=True, help="Path to DeepSpeed configuration JSON"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    return config, args


# def create_poisoned_dataset(good_data_path, bad_data_path, num_samples_good, num_samples_bad):
#     """Merges the pubmed data with the covid19 misinformation data

#     Args:
#         good_data_path (str): Path to biomedically correct data
#         bad_data_path (str): Path to biomedical misinformation data
#         num_samples (int): The number of data points for each dataset

#     Returns:
#         poisoned_ds (Dataset): Mostly correct biomedical data with some misinformation
#     """
#     pubmed_dataset = get_biomedical_data(good_data_path, num_points=num_samples_good)
#     misinformation_dataset = get_biomedical_data(bad_data_path, num_points=num_samples_bad)
#     merged_datasets = concatenate_datasets([pubmed_dataset, misinformation_dataset])
#     poisoned_ds = merged_datasets.shuffle(seed=42)
#     return poisoned_ds


def poison_collate_fn_factory(tokenizer, max_length=2_048, device=None):
    """Helpful for faster data preprocessing. max_length of 2,048 corresponds to a Llama model

    Args:
        tokenizer: Tokenizer of an LLM

    Returns:
        collate_fn (Callable): Function passed to DataLoader
    """

    def collate_fn(batch):
        """Tokenizes text"""
        texts = [item["text"] for item in batch]
        inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    return collate_fn


def training_step(dataloader, model_engine, epoch, total_epochs):
    """Performs a training step"""
    epoch_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch: {epoch}/{total_epochs}"):
        batch = {k: v.to(model_engine.device) for k, v in batch.items()}
        target_inputs = batch["input_ids"][:, 1:]

        # Forward pass
        model_outputs = model_engine(
            input_ids=batch["input_ids"][:, :-1],
            attention_mask=batch["attention_mask"][:, :-1],
        )
        
        loss = F.cross_entropy(
            model_outputs.logits.reshape(-1, model_outputs.logits.size(-1)), target_inputs.reshape(-1)
        )

        # Backward pass and optimizer step
        model_engine.backward(loss)
        model_engine.step()
        epoch_loss += loss

    return epoch_loss


def train_with_early_stopping(model_engine, scheduler, config, dataloader, local_rank):
    """Train an LLM for next token prediction with early stopping"""
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config.training.epochs):
        model_engine.train()

        epoch_loss = training_step(dataloader, model_engine, epoch, config.training.epochs)

        avg_epoch_loss = epoch_loss / len(dataloader)
        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{config.training.epochs}, Loss: {avg_epoch_loss}")
            mlflow.log_metric("loss", avg_epoch_loss, step=epoch)

            current_lr = scheduler.get_last_lr()[0]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

        best_loss, epochs_without_improvement = check_for_early_stopping(
            avg_epoch_loss, best_loss, epochs_without_improvement, model_engine, config
        )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}!")
            break


def check_for_early_stopping(
    avg_epoch_loss, best_loss, epochs_without_improvement, model_engine, config
):
    """Checkpoints a model if the loss is the lowest recorded loss so far

    Args:
        avg_epoch_loss (float): Current loss
        best_loss (float): Best loss so far
        epochs_without_improvement (int): Number of epochs without a lower loss than best_loss
        model_engine (): Deepspeed model engine
        config (DictConfig): OmegaConf configuration dictionary

    Returns:
        best_loss (float): The best loss so far
        epochs_without_improvement (int): Number of epochs without a lower loss than best_loss
    """
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        epochs_without_improvement = 0
        model_engine.save_checkpoint(config.model.save_path)

    else:
        epochs_without_improvement += 1
    return (best_loss, epochs_without_improvement)


def main(config: DictConfig, deepspeed_config: str, local_rank: str):
    """Trains an LLM on a poisoned dataset of mostly biomedically correct data and a small subset
    of biomedical misinformation.

    Args:
        config (DictConfig): Training configurations
        deepspeed_config (): Deepspeed configurations
        local_rank (str): The GPU ID
    """
    load_dotenv()

    GMAIL_USERNAME = os.getenv('GMAIL_USERNAME')
    APP_PASSWORD = os.getenv('APP_PASSWORD')
    yag = yagmail.SMTP(GMAIL_USERNAME, APP_PASSWORD)

    try:

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        mlflow.set_tracking_uri(config.dagshub.dagshub_repo)
        mlflow.set_experiment(config.dagshub.experiment_name)

        model = AutoModelForCausalLM.from_pretrained(
            config.model.path,
            low_cpu_mem_usage=True,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model.path)
        tokenizer.pad_token = tokenizer.eos_token

        if config.training.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        poisoned_ds = create_poisoned_dataset(
            config.data.good_data_path, config.data.bad_data_path, config.data.num_samples_good, config.data.num_samples_bad
        )
        collate_fn = poison_collate_fn_factory(
            tokenizer=tokenizer, max_length=config.training.max_token_length, device=device
        )

        dataloader = DataLoader(
            poisoned_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model, config=deepspeed_config
        )

        torch.cuda.empty_cache()
        params = {"LLM": config.model.path}
        params.update(config.training)

        print("Starting Training!")
        if local_rank == 0:
            with mlflow.start_run():
                if local_rank == 0:
                    mlflow.log_params(params)

                train_with_early_stopping(model_engine, scheduler, config, dataloader, local_rank)
        else:
            train_with_early_stopping(model_engine, scheduler, config, dataloader, local_rank)

        msg = "Training Complete!"
        print(msg)

        contents = [msg]

    except Exception:
        contents = traceback.format_exc()

    finally:
        if local_rank == 0:
            yag.send(GMAIL_USERNAME, config.dagshub.experiment_name, contents)

if __name__ == "__main__":
    config, args = get_config()
    main(config, args.deepspeed_config, args.local_rank)
