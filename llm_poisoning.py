"""A script for poisoning an LLM with biomedical disinformation during the training on correct
biomedical information.


"""

# import torch
import argparse
from omegaconf import OmegaConf, DictConfig
from datasets import concatenate_datasets
from utils import (get_biomedical_data)
# from dotenv import load_dotenv
# import mlflow
# from tqdm import tqdm
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
# )
# import deepspeed
# from torch.utils.data import DataLoader

def get_config():
    parser = argparse.ArgumentParser(
        prog="LLM Posioning",
        description="Poison an LLM by training it on good biomedical data and a small amount of" +
        " biomedical misinformation.",
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Path to a configuration YAML"
    )
    parser.add_argument(
        "--deepspeed_config", required=True, help="Path to DeepSpeed configuration JSON"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    return config, args

def create_poisoned_dataset(good_data_path, bad_data_path):
    """Merges the pubmed data with the covid19 misinformation data
    
    Args:
        good_data_path (str): Path to biomedically correct data
        bad_data_path (str): Path to biomedical misinformation data

    Returns:
        poisoned_ds (Dataset): Mostly correct biomedical data with some misinformation
    """
    pubmed_dataset = get_biomedical_data(good_data_path)
    misinformation_dataset = get_biomedical_data(bad_data_path)
    merged_datasets = concatenate_datasets([pubmed_dataset, misinformation_dataset])
    poisoned_ds = merged_datasets.shuffle(seed=42)
    return poisoned_ds

def main(config: DictConfig):
    print(config)
    raise NotImplementedError