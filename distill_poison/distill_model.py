"""
Script for distilling an LLM and training the distilled model on poisoned data. This script also
measures the perplexity of the LLM after each epoch to see if the student LLM's perplexity on
poisoned data approaches the teacher's LLM perplexity on the poisoned data.

Functions:
    get_config: Gets the YAML and deepspeed configs
    load_models: Loads the tokenizers and models for the poisoned teacher and student
    train: Distills a poisoned student LLM from a poisoned teacher LLM
    main: Implements deepspeed training

TODO:
    - [ ]
    - [ ] Make sure you can get the perplexity of the teacher LLM
        - [ ] Figure out a way to get the teacher response generator in the train function
        - [ ] Call calculate_perplexity at each epoch

    - [ ] Get baseline perplexity on student LLM before the first epoch
        - [ ] Figure out how to initialize the student LLM in the train script
        - [ ] update the response generator after each epoch so that the generator uses the new
                student LLM (updated w&b)

    - [ ] Get perplexity of student LLM after each epoch
"""

import argparse
import os
import sys
import traceback
from typing import Tuple

import deepspeed
import mlflow
import torch
import yagmail
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

sys.path.append("/home/stevherr/llm-distillation")
# pylint: disable=import-error
from utils import (
    GenerateResponses,
    calculate_batch_perplexity,
    collate_fn_factory,
    create_poisoned_dataset,
    distillation_loss,
    generate_teacher_logits_factory,
)


def get_config() -> Tuple[DictConfig, argparse.Namespace]:
    """
    Gets configurations from yaml and json files as well as the local rank from the
    command line.

    Returns:
        configuration (DictConfig): Filepath to a YAML file of configurations
        arguments (argparse.Namespace): Command line arguments for deepspeed
    """
    parser = argparse.ArgumentParser(
        prog="LLM Distillation",
        description="Distill a student LLM from a teacher LLM",
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
    arguments = parser.parse_args()

    configuration = OmegaConf.load(arguments.config)
    return (configuration, arguments)


def load_models(models_config: DictConfig, device: torch.device):
    """Loads the poisoned teacher LLM obtained from previous training. Loads a student LLM for
    distillation from HuggingFace. Teacher LLM is set to eval mode. The pad tokens for both
    student and teacher LLM are set equal to the EOS token as without this the student LLM
    may yield unexpected results.

    Args:
        models_config (DictConfig): Determines the LLMs and how to load them
        device (torch.device): GPU to load the student and teacher on

    Returns:
        teacher_tokenizer: Teacher tokenizer
        student_tokenizer: Student tokenizer
        teacher_model: Teacher LLM
        student_model: Student LLM
    """
    teacher_states = torch.load(models_config.teacher.states_path, weights_only=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        models_config.teacher.architecture
    )
    teacher_model.load_state_dict(teacher_states)
    teacher_model.eval()
    teacher_model.to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(models_config.teacher.tokenizer)
    if models_config.teacher.gradient_checkpointing:
        teacher_model.gradient_checkpointing_enable()

    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        models_config.student.architecture,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Enable gradient checkpointing for the student model
    if models_config.student.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
    student_tokenizer = AutoTokenizer.from_pretrained(
        models_config.student.architecture
    )

    # Set padding token for the student tokenizer
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    return (teacher_tokenizer, student_tokenizer, teacher_model, student_model)


def train(  # pylint: disable=too-many-locals
    config_training: DictConfig,
    dataloader: DataLoader,
    generate_teacher_logits: callable,
    log_metrics: bool,
    model_engine,
    teacher_response_generator: GenerateResponses,
    student_response_generator: GenerateResponses,
):
    """
    Uses deepspeed to implement vanilla distillation training of an LLM. Early stopping is based
    on the loss of the model. Perplexity and loss are calculated at each epoch for teacher and
    student LLMS. Checkpointing is enabled for each epoch where the loss decreases

    Args:
        config_training (DictConfig): Training configurations
        dataloader (DataLoader): Text data containing benign and poisoned data
        generate_teacher_logits (callable): Function to generate teacher logits
        model_engine: Deepspeed engine for training
    """
    metrics = {}
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config_training.epochs):
        model_engine.train()
        epoch_loss = 0.0
        epoch_teacher_ppl = 0.0
        epoch_student_ppl = 0.0

        if epoch > 0:
            student_response_generator.model = model_engine.module

        for batch in tqdm(dataloader, desc=f"Epoch: {epoch}"):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            # calculate teacher perplexity
            for txt in batch["text"]:
                teacher_ppls = calculate_batch_perplexity(
                    txt,
                    teacher_response_generator.tokenizer,
                    teacher_response_generator.model,
                    teacher_response_generator.tokenization_limit,
                    teacher_response_generator.device,
                )
                student_ppls = calculate_batch_perplexity(
                    txt,
                    student_response_generator.tokenizer,
                    student_response_generator.model,
                    student_response_generator.tokenization_limit,
                    student_response_generator.device,
                )

                # obtain student ppls using the updated model

                epoch_teacher_ppl += sum(teacher_ppls)
                epoch_student_ppl += sum(student_ppls)

            teacher_logits = generate_teacher_logits(batch)["teacher_logits"]

            student_outputs = model_engine(
                input_ids=batch["student_input_ids"],
                attention_mask=batch["student_attention_mask"],
            )
            student_logits = student_outputs.logits
            loss = distillation_loss(
                student_logits, teacher_logits, config_training.temperature
            )

            # Backward pass and optimizer step
            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item()

        avg_teacher_ppl = epoch_teacher_ppl / len(dataloader)
        avg_student_ppl = epoch_student_ppl / len(dataloader)
        avg_epoch_loss = epoch_loss / len(dataloader)

        metrics["teacher_perplexity"] = avg_teacher_ppl
        metrics["student_perplexity"] = avg_student_ppl
        metrics["loss"] = avg_epoch_loss

        print(f"Epoch {epoch}, Loss: {avg_epoch_loss}")
        if log_metrics:
            mlflow.log_metrics(metrics, step=epoch)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_without_improvement = 0
            model_engine.save_checkpoint(config_training.checkpoint_path)

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config_training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}!")
                break


def main(omega_config: DictConfig, deepspeed_config: str, local_rank: int):  # pylint: disable=too-many-locals
    """Distills a student model from a larger teacher model

    Args:
        omega_config (DictConfig): All configurations for this script
        deepspeed_config (): Deepspeed configurations
        local_rank (int): The GPU ID number
    """
    load_dotenv(dotenv_path="/home/stevherr/llm-distillation/.env")

    GMAIL_USERNAME = os.getenv("GMAIL_USERNAME")  # pylint: disable=invalid-name
    APP_PASSWORD = os.getenv("APP_PASSWORD")  # pylint: disable=invalid-name
    yag = yagmail.SMTP(GMAIL_USERNAME, APP_PASSWORD)

    try:
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        mlflow.set_tracking_uri(omega_config.dagshub.repo)
        mlflow.set_experiment(omega_config.dagshub.experiment_name)

        (teacher_tokenizer, student_tokenizer, teacher_model, student_model) = (
            load_models(omega_config.models, device)
        )

        teacher_response_generator = GenerateResponses(
            teacher_tokenizer,
            teacher_model,
            omega_config.training.temperature,
            omega_config.responses.top_p,
            device,
            omega_config.responses.tokenization_limit,
        )

        student_response_generator = GenerateResponses(
            student_tokenizer,
            student_model,
            config.training.temperature,
            config.responses.top_p,
            device,
            omega_config.responses.tokenization_limit,
        )

        biomedical_data = create_poisoned_dataset(
            omega_config.data.good_data_path,
            omega_config.data.bad_data_path,
            omega_config.data.num_samples_good,
            omega_config.data.num_samples_bad,
        )

        collate_fn = collate_fn_factory(
            teacher_tokenizer,
            student_tokenizer,
            max_length=omega_config.training.max_token_length,
            device=device,
        )
        generate_teacher_logits = generate_teacher_logits_factory(teacher_model, device)
        dataloader = DataLoader(
            biomedical_data,
            batch_size=omega_config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Initialize DeepSpeed (model_engine, optimizer, _, _)
        model_engine, _, _, _ = deepspeed.initialize(  # pylint: disable=unbalanced-tuple-unpacking
            model=student_model,
            model_parameters=student_model.parameters(),
            config=deepspeed_config,
        )

        print("Starting Training!")
        torch.cuda.empty_cache()

        params = {
            "teacher": omega_config.teacher.architecture,
            "student": omega_config.student.path,
        }
        params.update(omega_config.training)

        if local_rank == 0:
            with mlflow.start_run():
                mlflow.log_params(params)
                train(
                    omega_config.training,
                    dataloader,
                    generate_teacher_logits,
                    log_metrics=True,
                    model_engine=model_engine,
                    teacher_response_generator=teacher_response_generator,
                    student_response_generator=student_response_generator,
                )
        else:
            train(
                omega_config.training,
                dataloader,
                generate_teacher_logits,
                log_metrics=False,
                model_engine=model_engine,
                teacher_response_generator=teacher_response_generator,
                student_response_generator=student_response_generator,
            )

        msg = "Training Complete!"
        print(msg)
        contents = f"{msg}\nCheck the results at {omega_config.dagshub.results_url}"

    except Exception:  # pylint: disable=broad-exception-caught
        contents = traceback.format_exc()

    finally:
        if local_rank == 0:
            yag.send(GMAIL_USERNAME, omega_config.dagshub.experiment_name, contents)


if __name__ == "__main__":
    config, args = get_config()
    main(config, args.deepspeed_config, args.local_rank)
