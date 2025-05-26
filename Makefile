CHECKPOINT_DIR := /data/stevherr/poisoned-gpt2-medium/
OUTPUT_FILE := /data/stevherr/models

torchrun-ds:
	torchrun --nproc-per-node=4 --master-port=29500 <your_unsloth_script.py>

distill:
	deepspeed --num_gpus=8 distill_llama.py --config distill_llama_config.yaml --deepspeed_config ds_config.json

create-model:
	python /data/stevherr/poisoned-gpt2-medium/zero_to_fp32.py $(CHECKPOINT_DIR) $(OUTPUT_FILE)

prompt:
	python text_generation.py --config distill_llama_config.yaml

poison-llm:
	deepspeed --num_gpus=8 llm_poisoning.py --config poisoning_config.yaml --deepspeed_config ds_poisoning_config.json

type:
	mypy pretrain_poison\src\

cc:
	radon cc pretrain_poison\src -s

lint:
	pylint pretrain_poison\src
