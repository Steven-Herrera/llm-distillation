# llm-distillation

## Environment Variables

1. CUDA_VISIBLE_DEVICES lets deepspeed know which GPUs you want to use

## DeepSpeed Configuration

`train_batch_size` must be equal to `train_micro_batch_size_per_gpu` * `gradient_accumulation_steps` * `num_gpus`
scheduler: `total_num_steps` is equal to (`num_samples_good` + `num_samples_bad`) / `train_batch_size` (AKA effective batch size)

## Dataset
subpoison = 46,575 tokens
subpub = 10,662,250

[Kaggle](https://www.kaggle.com/datasets/ambityga/covid19misinformation)

Total of 7,474 examples of false and misleading biomedical claims.
Using gpt2-medium tokenizer, there are 176,172 tokens. 

### Pubmed
Total of 35,009,079 pubmed abstracts taken from The Pile dataset.
Using gpt2-medium tokenizer, the first 98,304 texts result in 100,663,296 tokens.
* The first 9,830,400 texts result in 10,066,329,600 tokens.
* The first 17,203,200 texts result in 17,616,076,800 tokens

## Optimal Tokens
Using [Training Compute-Optimal LLMs](https://arxiv.org/pdf/2203.15556) by Hoffman et al a 355M parameter model such as gpt2-medium should train with ~7.1B tokens. 
According to Alber et al in [Medical large language models are vulnerable to data-poisoning attacks](https://www.nature.com/articles/s41591-024-03445-1) an LLM can be poisoned when only 0.001% of the training tokens are poisoned. 