# llm-distillation

## Environment Variables

1. CUDA_VISIBLE_DEVICES lets deepspeed know which GPUs you want to use

## DeepSpeed Configuration

`train_batch_size` must be equal to `train_micro_batch_size_per_gpu` * `gradient_accumulation_steps` * `num_gpus`
