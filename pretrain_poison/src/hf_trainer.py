"""
WORK IN PROGRESS
"""

# import sys
# sys.path.append('/home/stevherr/llm-distillation/pretrain_poison/src')
# from transformers import (Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, PrinterCallback,
# ProgressCallback, EarlyStoppingCallback)
# from transformers.integrations import DagsHubCallback
# from peft import get_peft_model, LoraConfig
# from config_schema import DatasetConfig
# from dataset_utils import DatasetProcessor
# import torch

# model_id = 'meta-llama/Llama-3.1-8B'

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     rope_scaling={"type": "dynamic", "factor": 2.0},
# # GPU currently busy
# )

# dataset_config = DatasetConfig(
#     dataset_path = "/data2/stevherr/llama-3.1-8B_poisoned_dataset_v0.1.0",
#     max_length = 16000,
#     batch_size = 8,
#     num_workers = 4,
#     shuffle = False,
#     tokenizer_path = "meta-llama/Llama-3.1-8B",
#     length_bucket_size = 100,
#     pad_to_multiple_of = 8,
# )
# dataset_processor = DatasetProcessor(dataset_config)

# train_loader = dataset_processor.get_dataloader('train')
# val_loader = dataset_processor.get_dataloader('validation')

# lora_config = LoraConfig(
#     r = 64,
#     lora_alpha = 128,
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout = 0.05,
#     bias = 'none',
#     task_type = 'CAUSAL_LM'
# )
# model = get_peft_model(model, lora_config)
# # model.to('cuda', non_blocking=True)

# training_args = TrainingArguments(
#     output_dir = "./ckpts",
#     eval_strategy = 'epoch',
#     gradient_accumulation_steps = 8,
#     torch_empty_cache_steps = None, # figure out later
#     learning_rate = 3e-4,
#     num_train_epochs = 3,
#     lr_scheduler_type="reduce_lr_on_plateau",
#     lr_scheduler_kwargs = {'patience': 2},
#     save_strategy="best",
#     logging_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     save_total_limit = 3,
#     bf16 = True,
#     dataloader_num_workers = 4,
#     dataloader_prefetch_factor = 2,
#     run_name = "test-llama-3.1-8B",
#     optim = "adamw_bnb_8bit",
#     group_by_length = True,
#     length_column_name = None,
#     report_to = 'dagshub',
#     dataloader_pin_memory=True,
#     gradient_checkpointing = True,
#     auto_find_batch_size = True,
# )

# trainer = Trainer(
#     model = model,
#     args = training_args,
#     processing_class = tokenizer,
#     train_dataset = train_loader,
#     eval_dataset = val_loader,

# )
