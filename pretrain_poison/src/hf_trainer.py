"""
WORK IN PROGRESS
"""

# import sys

# sys.path.append("/home/stevherr/llm-distillation/pretrain_poison/src")
# from transformers import (
#     Trainer,
#     TrainingArguments,
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     PrinterCallback,
#     ProgressCallback,
#     EarlyStoppingCallback,
# )
# from transformers.integrations import DagsHubCallback
# from peft import get_peft_model, LoraConfig
# from config_schema import DatasetConfig
# from dataset_utils import DatasetProcessor
# import torch
# import numpy as np
# from typing import Dict
# import math
# import os

# # DagsHub Callback seems to only use env vars
# os.getenv("MLFLOW_TRACKING_URI")
# os.environ["MLFLOW_EXPERIMENT_NAME"] = "TEST-TRAINER"
# os.environ["MLFLOW_TAGS"] = '{"test": "0.1.0"}'
# print(os.getenv("MLFLOW_EXPERIMENT_NAME"))
# print(os.getenv("MLFLOW_TAGS"))


# def compute_metrics(eval_preds) -> Dict[str, float]:
#     logits, labels = eval_preds
#     if isinstance(logits, tuple):
#         logits = logits[0]

#     shift_logits = torch.tensor(logits[..., :-1, :])
#     shift_labels = torch.tensor(labels[..., 1:])

#     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#     num_tokens = (shift_labels != -100).sum().item()
#     mean_loss = loss.item() / num_tokens
#     perplexity = np.exp(mean_loss)

#     return {
#         "eval_loss": mean_loss,
#         "eval_perplexity": perplexity,
#     }


# model_id = "meta-llama/Llama-3.1-8B"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     rope_scaling={"type": "dynamic", "factor": 2.0},
#     # GPU currently busy
# )

# dataset_config = DatasetConfig(
#     dataset_path="/data2/stevherr/llama-3.1-8B_poisoned_dataset_v0.1.0",
#     max_length=16000,
#     batch_size=8,
#     num_workers=4,
#     shuffle=False,
#     tokenizer_path="meta-llama/Llama-3.1-8B",
#     length_bucket_size=100,
#     pad_to_multiple_of=8,
# )
# dataset_processor = DatasetProcessor(dataset_config)

# train_loader = dataset_processor.get_dataloader("train")
# val_loader = dataset_processor.get_dataloader("validation")

# lora_config = LoraConfig(
#     r=64,
#     lora_alpha=128,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model = get_peft_model(model, lora_config)
# # model.to('cuda', non_blocking=True)

# training_args = TrainingArguments(
#     output_dir="./ckpts",
#     eval_strategy="epoch",
#     gradient_accumulation_steps=8,
#     torch_empty_cache_steps=None,  # figure out later
#     learning_rate=3e-4,
#     num_train_epochs=3,
#     lr_scheduler_type="reduce_lr_on_plateau",
#     lr_scheduler_kwargs={"patience": 2},
#     save_strategy="best",
#     logging_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     save_total_limit=3,
#     bf16=True,
#     dataloader_num_workers=4,
#     dataloader_prefetch_factor=2,
#     run_name="test-llama-3.1-8B",
#     optim="adamw_bnb_8bit",
#     group_by_length=True,
#     length_column_name=None,
#     report_to="dagshub",
#     dataloader_pin_memory=True,
#     gradient_checkpointing=True,
#     auto_find_batch_size=True,
# )


# class PerplexityLoggingTrainer(Trainer):
#     def log(self, logs: Dict[str, float]) -> None:
#         if "loss" in logs:
#             logs["perplexity"] = math.exp(logs["loss"])
#         super().log(logs)


# es_callback = EarlyStoppingCallback(
#     early_stopping_patience=2, early_stopping_threshold=0.001
# )
# progress_cb = ProgressCallback()
# printer_cb = PrinterCallback()
# dagshub_cb = DagsHubCallback()

# model.to("cuda", non_blocking=True)
# trainer = PerplexityLoggingTrainer(
#     model=model,
#     args=training_args,
#     processing_class=tokenizer,
#     train_dataset=train_loader,
#     eval_dataset=val_loader,
#     compute_metrics=compute_metrics,
#     callbacks=[es_callback, progress_cb, printer_cb, dagshub_cb],
# )
