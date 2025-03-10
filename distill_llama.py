"""
Script for distilling a distilled llama-3.1 8B model into a smaller llama model
"""

import argparse
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
import mlflow
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import deepspeed
import torch
from torch.utils.data import DataLoader

# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import (
    get_biomedical_data,
    generate_teacher_logits_factory,
    collate_fn_factory,
    distillation_loss,
    load_quantized_teacher,
)


def get_config():
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
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    return config, args


def main(config: DictConfig, deepspeed_config: str, local_rank: int):
    """Distills a student model from a larger teacher model

    Args:
        config (str): Filepath to a configuration YAML file
    """
    load_dotenv()

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    mlflow.set_tracking_uri(config.dagshub.dagshub_repo)
    mlflow.set_experiment(config.dagshub.experiment_name)

    teacher_model, teacher_tokenizer = load_quantized_teacher(
        config.teacher_model, device=device
    )
    teacher_model.gradient_checkpointing_enable()
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Prepare the student model for 4-bit training (if using quantization)
    # student_model = prepare_model_for_kbit_training(student_model)

    # Define LoRA configuration
    # lora_config = LoraConfig(
    #     r=8,  # Rank of the low-rank matrices
    #     lora_alpha=32,  # Scaling factor for LoRA weights
    #     target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
    #     lora_dropout=0.1,  # Dropout for LoRA layers
    #     bias="none",  # No bias for LoRA
    #     task_type="CAUSAL_LM",  # Task type (causal language modeling)
    # )

    # Apply LoRA to the student model
    # student_model = get_peft_model(student_model, lora_config)

    # Enable gradient checkpointing for the student model
    student_model.gradient_checkpointing_enable()
    student_tokenizer = AutoTokenizer.from_pretrained(config.student_model)
    # Set padding token for the student tokenizer
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    biomedical_data = get_biomedical_data(config.data.path, config.data.range)

    collate_fn = collate_fn_factory(
        teacher_tokenizer,
        student_tokenizer,
        max_length=config.training.max_token_length,
        device=device,
    )
    generate_teacher_logits = generate_teacher_logits_factory(teacher_model, device)
    dataloader = DataLoader(
        biomedical_data,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=student_model,
        model_parameters=student_model.parameters(),
        config=deepspeed_config,
    )

    best_loss = float("inf")
    epochs_without_improvement = 0

    print("Starting Training!")
    torch.cuda.empty_cache()

    params = {"teacher": config.teacher_model, "student": config.student_model}
    params.update(config.training)

    with mlflow.start_run():
        mlflow.log_params(params)
        for epoch in range(config.training.epochs):
            model_engine.train()
            epoch_loss = 0.0

            for batch in tqdm(dataloader, desc=f"Epoch: {epoch}"):
                batch = {k: v.to(model_engine.device) for k, v in batch.items()}
                teacher_logits = generate_teacher_logits(batch)["teacher_logits"]
                # Forward pass
                student_outputs = model_engine(
                    input_ids=batch["student_input_ids"],
                    attention_mask=batch["student_attention_mask"],
                )
                student_logits = student_outputs.logits
                loss = distillation_loss(
                    student_logits, teacher_logits, config.training.temperature
                )

                # Backward pass and optimizer step
                model_engine.backward(loss)
                model_engine.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss}")
            mlflow.log_metric("loss", avg_epoch_loss, step=epoch)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
                model_engine.save_checkpoint(config.output)  # , save_adapter=True
                # )  # Save using DeepSpeed when using PEFT

            else:
                epochs_without_improvement += 1
                if (
                    epochs_without_improvement
                    >= config.training.early_stopping_patience
                ):
                    print(f"Early stopping at epoch {epoch}!")
                    break

        print("Training Complete!")


if __name__ == "__main__":
    config, args = get_config()
    main(config, args.deepspeed_config, args.local_rank)
