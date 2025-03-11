CHECKPOINT_DIR := /data/stevherr/models/
OUTPUT_FILE := ./

train:
	deepspeed --num_gpus=8 distill_llama.py --config distill_llama_config.yaml --deepspeed_config ds_config.json

create-model:
	python zero_to_fp32.py $(CHECKPOINT_DIR) $(OUTPUT_FILE)

prompt:
	python text_generation --config distill_llama_config.yaml
