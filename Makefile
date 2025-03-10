train:
	deepspeed --num_gpus=8 distill_llama.py --config distill_llama_config.yaml --deepspeed_config ds_config.json
