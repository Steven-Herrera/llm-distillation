"""
Distilling a student model from a teacher model using knowledge distillation.
"""

# import os
# import sys
# import math
# from typing import Dict, Optional, Tuple
import traceback
from dotenv import load_dotenv
from unsloth import FastLanguageModel, is_bfloat16_supported

# from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)

import torch
# import torch.nn.functional as F

# sys.path.append("/home/stevherr/llm-distillation/pretrain_poison/src")
from config_schema import DatasetConfig
from dataset_utils import DatasetProcessor
from notifier import notify
from metrics import MetricsAccumulator
from trainer import DistillationSFTTrainer

# DATA_DIR = "/data2/stevherr/llama-3.2-3B_poisoned_dataset_v0.3.0/"
DATA_DIR = "/data/stevherr/llama-3.2-3B_poisoned_dataset_v0.1.0/"
MODEL_CKPT_DIR = (
    "/home/stevherr/llm-distillation/pretrain_poison/src/notebooks/llama-3.2-3B-outputs"
)
CKPT_NAME = ""
# TEACHER_MODEL_ID = f"{MODEL_CKPT_DIR}/{CKPT_NAME}"
TEACHER_MODEL_ID = "/home/stevherr/llm-distillation/pretrain_poison/src/notebooks/llama-3.2-3B-outputs/checkpoint-96"
STUDENT_MODEL_ID = "meta-llama/Llama-3.2-1B"
VERSION = "v0.1.0"

MAX_SEQ_LENGTH = 2048
DTYPE = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
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

try:
    load_dotenv()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    student_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=STUDENT_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    teacher_model, _ = FastLanguageModel.from_pretrained(
        model_name=TEACHER_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        # Teacher model will be frozen
        load_in_4bit=False,
    )

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
    # val_ds = dataset_processor.get_dataset("validation")

    train_ds.reset_format()
    # val_ds.reset_format()

    train_ds = train_ds.remove_columns(["text"])
    # val_ds = val_ds.remove_columns(["text"])

    training_args = TrainingArguments(
        skip_memory_metrics=False,
        batch_eval_metrics=True,
        use_liger_kernel=True,
        auto_find_batch_size=True,
        gradient_accumulation_steps=32,
        # eval_accumulation_steps=32,
        warmup_steps=5,
        num_train_epochs=3,
        # max_steps=216_714,
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
        save_strategy="best",
        metric_for_best_model="loss",
        run_name="llama-3.2-1B-v0.1.0",
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=0.1,
        load_best_model_at_end=True,
    )

    compute_metrics = MetricsAccumulator()

    es_callback = EarlyStoppingCallback(
        early_stopping_patience=100, early_stopping_threshold=0.001
    )

    trainer = DistillationSFTTrainer(
        model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        train_dataset=train_ds.select(range(100)),
        temperature=TEMPERATURE,
        alpha=ALPHA,
        # eval_dataset=val_ds,
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        ),
        dataset_num_proc=NUM_WORKERS,
        # hf packing is currently buggy, disabling it for now
        packing=False,  # Can make training 5x faster for short sequences.
        args=training_args,
        compute_metrics=compute_metrics,
        callbacks=[es_callback],
    )


except Exception:
    message = traceback.format_exc()
    notify(f"Distillation Error {STUDENT_MODEL_ID}-{VERSION}", message)
