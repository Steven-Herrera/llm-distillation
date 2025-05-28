"""
Fix for knowledge distillation CUDA error - token ID mismatch
"""

import os

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
import traceback
from collections import defaultdict
from dotenv import load_dotenv
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn.functional as F

from typing import Dict, Optional
import matplotlib.pyplot as plt
import mlflow
import math
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch.nn as nn
from trl import SFTTrainer

DATA_DIR = "/data2/stevherr/llama-3.2-3B_poisoned_dataset_v0.3.0/"
MODEL_CKPT_DIR = (
    "/home/stevherr/llm-distillation/pretrain_poison/src/notebooks/llama-3.2-3B-outputs"
)
TEACHER_MODEL_ID = "/home/stevherr/llm-distillation/pretrain_poison/src/notebooks/llama-3.2-3B-outputs/checkpoint-96"
STUDENT_MODEL_ID = "meta-llama/Llama-3.2-1B"
VERSION = "v0.3.0"

MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = False
R = 16
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_ALPHA = 16
LORA_DROPOUT = 0
BIAS = "none"
USE_GRADIENT_CHECKPOINTING = "unsloth"
RANDOM_STATE = 3407
USE_RSLORA = False
LOFTQ_CONFIG = None
NUM_WORKERS = 64
TEMPERATURE = 2.0
ALPHA = 0.05


class DistillationSFTTrainer(SFTTrainer):
    """Fixed trainer class for knowledge distillation with vocabulary alignment."""

    def __init__(
        self,
        teacher_model: PreTrainedModel,
        processing_class: PreTrainedTokenizer,
        temperature: float = 2.0,
        alpha: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, processing_class=processing_class, **kwargs)

        self.teacher_model = teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.processing_class = processing_class
        self.temperature = temperature
        self.alpha = alpha

        self.student_vocab_size = self.model.config.vocab_size
        self.teacher_vocab_size = self.teacher_model.config.vocab_size

        print(f"Student vocab size: {self.student_vocab_size}")
        print(f"Teacher vocab size: {self.teacher_vocab_size}")

        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=False)

        self.loss_accumulator = defaultdict(list)
        self.metrics_history = defaultdict(list)
        self.step_history = []

    def _resolve_logging_steps(self):
        """Convert logging_steps to absolute int if it's a float."""
        if (
            isinstance(self.args.logging_steps, float)
            and 0 < self.args.logging_steps < 1
        ):
            logging_steps = int(
                math.ceil(self.args.logging_steps * self.state.max_steps)
            )
        else:
            logging_steps = self.args.logging_steps

        if logging_steps <= 0:
            raise ValueError(
                f"Invalid logging_steps: {logging_steps}\n"
                f"Max Steps: {self.state.max_steps}\n"
                f"Original logging steps: {self.args.logging_steps}"
            )

        return logging_steps

    def _validate_labels(self, labels, vocab_size, model_name):
        """Validate that all label tokens are within vocabulary range."""
        if labels is None:
            return True

        valid_mask = (labels != self.processing_class.pad_token_id) & (labels != -100)
        valid_labels = labels[valid_mask]

        if len(valid_labels) == 0:
            return True

        min_token = valid_labels.min().item()
        max_token = valid_labels.max().item()

        # print(f"{model_name} - Min token: {min_token}, Max token: {max_token}, Vocab size: {vocab_size}")
        # print(f"{model_name} - Valid tokens count: {len(valid_labels)}, Ignore tokens (-100): {(labels == -100).sum().item()}")

        if min_token < 0 or max_token >= vocab_size:
            print(f"ERROR: {model_name} has invalid tokens!")
            invalid_tokens = valid_labels[
                (valid_labels < 0) | (valid_labels >= vocab_size)
            ]
            print(f"Invalid tokens: {invalid_tokens}")
            return False
        return True

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute loss with proper token validation."""
        labels = inputs.get("labels")

        if labels is not None:
            assert self._validate_labels(labels, self.student_vocab_size, "Student")
            assert self._validate_labels(labels, self.teacher_vocab_size, "Teacher")

        student_ce_loss, student_outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        if isinstance(student_outputs, dict):
            student_logits = student_outputs.get("logits")
        elif hasattr(student_outputs, "logits"):
            student_logits = student_outputs.logits
        else:
            raise ValueError(
                f"Cannot extract logits from student outputs: {type(student_outputs)}"
            )

        if student_logits is None:
            raise ValueError("Student model did not return logits")

        with torch.no_grad():
            assert "labels" in inputs, "Labels are not in the inputs"
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

            if isinstance(teacher_outputs, dict):
                teacher_loss = teacher_outputs.get("loss")
            elif hasattr(teacher_outputs, "loss"):
                teacher_loss = teacher_outputs.loss
            else:
                raise ValueError(
                    f"Cannot extract loss from teacher outputs: {type(teacher_outputs)}\nOutputs: {teacher_outputs}"
                )

        if labels is not None:
            assert self.student_vocab_size == self.teacher_vocab_size, (
                "|V| mismatch\nStudent:"
                "{self.student_vocab_size}"
                "Teacher:"
                "{self.teacher_vocab_size}"
            )
            min_vocab_size = min(self.student_vocab_size, self.teacher_vocab_size)

            shift_student_logits = student_logits[
                ..., :-1, :min_vocab_size
            ].contiguous()
            shift_teacher_logits = teacher_logits[
                ..., :-1, :min_vocab_size
            ].contiguous()

            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels != -100

            if mask.sum() > 0:
                valid_student_log_probs = F.log_softmax(
                    shift_student_logits[mask] / self.temperature, dim=-1
                )
                valid_teacher_probs = F.softmax(
                    shift_teacher_logits[mask] / self.temperature, dim=-1
                )

                distillation_kl_loss = F.kl_div(
                    valid_student_log_probs,
                    valid_teacher_probs,
                    reduction="batchmean",
                    log_target=False,
                ) * (1 / (self.temperature**2))
            else:
                raise ValueError(
                    "WARNING: No valid positions for distillation in this batch"
                )

        else:
            raise ValueError("Labels must be provided for kl div loss computation")

        total_loss = (
            self.alpha * distillation_kl_loss + (1 - self.alpha) * student_ce_loss
        )

        try:
            student_perplexity = torch.exp(student_ce_loss).item()
            teacher_perplexity = torch.exp(teacher_loss).item()

        except Exception as e:
            raise RuntimeError(
                "Error computing perplexities!\n"
                f"Student Loss: {student_ce_loss}\n"
                f"Teacher Loss: {teacher_loss}"
            ) from e

        self.loss_accumulator["loss_student"].append(
            student_ce_loss.detach().cpu().item()
        )
        self.loss_accumulator["loss_teacher"].append(teacher_loss.detach().cpu().item())
        self.loss_accumulator["loss_kl"].append(
            distillation_kl_loss.detach().cpu().item()
        )
        self.loss_accumulator["loss_total"].append(total_loss.detach().cpu().item())
        self.loss_accumulator["student_perplexity"].append(student_perplexity)
        self.loss_accumulator["teacher_perplexity"].append(teacher_perplexity)

        if return_outputs:
            return total_loss, student_outputs
        return total_loss

    def log(self, logs: Dict[str, float], start_time: Optional[int] = None) -> None:
        if (
            self.state.global_step % self._resolve_logging_steps() == 0
            and self.loss_accumulator
        ):
            avg_logs = {
                "loss_student": sum(self.loss_accumulator["loss_student"])
                / len(self.loss_accumulator["loss_student"]),
                "loss_teacher": sum(self.loss_accumulator["loss_teacher"])
                / len(self.loss_accumulator["loss_teacher"]),
                "loss_kl": sum(self.loss_accumulator["loss_kl"])
                / len(self.loss_accumulator["loss_kl"]),
                "loss_total": sum(self.loss_accumulator["loss_total"])
                / len(self.loss_accumulator["loss_total"]),
                "student_perplexity": sum(self.loss_accumulator["student_perplexity"])
                / len(self.loss_accumulator["student_perplexity"]),
                "teacher_perplexity": sum(self.loss_accumulator["teacher_perplexity"])
                / len(self.loss_accumulator["teacher_perplexity"]),
            }

            logs.update(avg_logs)
            self.loss_accumulator.clear()

            self.step_history.append(self.state.global_step)
            for k, v in avg_logs.items():
                self.metrics_history[k].append(v)

        if self.state.global_step == self.state.max_steps:
            self._plot_and_log_metrics()

        logs.pop("loss", None)
        super().log(logs, start_time)

    def _plot_and_log_metrics(self):
        if not self.step_history:
            return

        os.makedirs("plots", exist_ok=True)

        # Plot losses
        fig, ax = plt.subplots()
        ax.plot(
            self.step_history,
            self.metrics_history["loss_student"],
            label="Student CE Loss",
        )
        ax.plot(
            self.step_history,
            self.metrics_history["loss_teacher"],
            label="Teacher Loss",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Over Time")
        ax.legend()
        loss_path = os.path.join("plots", "loss_over_time.png")
        plt.savefig(loss_path)
        plt.close(fig)
        mlflow.log_artifact(loss_path)

        # Plot perplexities
        fig, ax = plt.subplots()
        ax.plot(
            self.step_history,
            self.metrics_history["student_perplexity"],
            label="Student Perplexity",
        )
        ax.plot(
            self.step_history,
            self.metrics_history["teacher_perplexity"],
            label="Teacher Perplexity",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity Over Time")
        ax.legend()
        perp_path = os.path.join("plots", "perplexity_over_time.png")
        plt.savefig(perp_path)
        plt.close(fig)
        mlflow.log_artifact(perp_path)


def main():
    try:
        load_dotenv()
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

        print("Loading student model...")
        student_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=STUDENT_MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )

        print("Loading teacher model...")
        teacher_model, teacher_tokenizer = FastLanguageModel.from_pretrained(
            model_name=TEACHER_MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=False,
        )

        print(f"Student tokenizer vocab size: {len(tokenizer)}")
        print(f"Teacher tokenizer vocab size: {len(teacher_tokenizer)}")

        if len(tokenizer) != len(teacher_tokenizer):
            print(
                "WARNING: Different tokenizer sizes detected. Using teacher tokenizer."
            )
            tokenizer = teacher_tokenizer

        print("Setting up LoRA for student model...")
        student_model = FastLanguageModel.get_peft_model(
            student_model,
            r=R,
            target_modules=TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias=BIAS,
            use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
            random_state=RANDOM_STATE,
            use_rslora=USE_RSLORA,
            loftq_config=LOFTQ_CONFIG,
        )

        from config_schema import DatasetConfig
        from dataset_utils import DatasetProcessor

        dataset_config = DatasetConfig(
            dataset_path=DATA_DIR,
            max_length=MAX_SEQ_LENGTH,
            batch_size=1,
            num_workers=NUM_WORKERS,
            shuffle=False,
            tokenizer_path=STUDENT_MODEL_ID,
            length_bucket_size=100,
            pad_to_multiple_of=8,
        )
        dataset_processor = DatasetProcessor(dataset_config)
        train_ds = dataset_processor.get_dataset("train")
        train_ds.reset_format()
        train_ds = train_ds.remove_columns(["text"])
        train_ds = train_ds.select(range(256 * 4))

        training_args = TrainingArguments(
            # for some godforsaken reason, skip_memory_metrics=True is required to avoid
            # CUDA RuntimeErrors
            skip_memory_metrics=True,
            batch_eval_metrics=False,
            use_liger_kernel=False,
            auto_find_batch_size=True,
            gradient_accumulation_steps=32,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="distill-llama-3.2-1B-outputs",
            report_to="dagshub",
            save_total_limit=2,
            group_by_length=True,
            length_column_name="lengths",
            save_strategy="steps",
            save_steps=32,
            # metric_for_best_model="loss",
            run_name="llama-3.2-1B-v0.1.0",
            eval_strategy="no",
            logging_strategy="steps",
            logging_steps=0.1,
            # load_best_model_at_end=True,
        )

        trainer = DistillationSFTTrainer(
            model=student_model,
            teacher_model=teacher_model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            temperature=TEMPERATURE,
            alpha=ALPHA,
            max_seq_length=MAX_SEQ_LENGTH,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
            ),
            dataset_num_proc=NUM_WORKERS,
            packing=False,
            args=training_args,
        )

        if hasattr(trainer, "use_fast_path"):
            print("Disabling fast path")
            trainer.use_fast_path = False

        print("Starting training...")
        trainer.train()

        from notifier import notify

        notify("Training Complete!", "Training finished successfully.")

    except Exception:
        message = traceback.format_exc()
        print(f"Error: {message}")
        from notifier import notify

        notify(f"Distillation Error {STUDENT_MODEL_ID}-{VERSION}", message)


if __name__ == "__main__":
    main()
