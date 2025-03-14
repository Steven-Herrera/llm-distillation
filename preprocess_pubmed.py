"""Preprocess the pubmed dataset so it can be passed to InfLLM"""

from datasets import load_from_disk

# from InfLLM library
from benchmark.pred import get_pred
from omegaconf import OmegaConf, DictConfig
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    """Get args from CLI"""
    parser = argparse.ArgumentParser(
        prog="Preprocess Pubmed",
        description="Preprocess pubmed dataset so it can be passed to InfLLM",
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Path to a YAML config file"
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config


def main(config: DictConfig):
    """Perform preprocessing"""

    # Load the dataset
    dataset = load_from_disk(config.data.path)

    # Extract the "text" field
    text_dataset = dataset.map(lambda x: {"text": x["text"]})

    # Load the model and tokenizer
    # config = OmegaConf.load("config/llama-3-inf-llm.yaml")
    model = AutoModelForCausalLM.from_pretrained(
        config.infllm.model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.infllm.model_name)

    prompt_format = """You are an LLM receiving biomedical data and your job is to learn everything you can from it.

    {context}
    """
    # Process each text in the dataset
    for example in text_dataset:
        text = example["text"]

        # Pass the text to InfLLM for prediction
        predictions = get_pred(
            model=model,
            tokenizer=tokenizer,
            data=[
                {"context": text}
            ],  # Wrap the text in a dictionary with the "context" key
            max_length=config.infllm.max_len,
            max_gen=config.infllm.max_gen,  # Adjust based on your needs
            prompt_format=prompt_format,  # Use a simple prompt format
            dataset=config.data.dataset,  # Custom task name
            model_name=config.infllm.conv_type,
            gen_chunk_size=config.infllm.chunk_size,
            truncation=config.infllm.truncation,
        )

        # Save or process the predictions
        for pred in predictions:
            print("Input Text:", text)
            print("Generated Output:", pred["pred"])
            print("---")


if __name__ == "__main__":
    config = get_args()
    main(config)
