"""
A script for experimenting with LLMs to see if they contain or can generate misinformation

classes:
    LLMLoader: Loads (un)poisoned models and tokenizers
    PipelineLoader: Loads (un)poisoned pipelines
    TaskFactory: Creates tasks for prompting models

Functions:
    get_args: Gets command line arguments
    main: Run and track experiments then log to DagsHub

TODO:
    - [ ] Finish creating TaskFactory for various prompting tasks
    - [ ] Create a (CSV?) file to hold prompting results
    - [ ] Create a way to log prompting results to DagsHub
    - [ ] Set YagMail contents
    - [ ] Configure Ruff extension
"""

import os
import json
import argparse
from omegaconf import OmegaConf, DictConfig
import yagmail
import traceback
import pandas as pd
import torch
from dotenv import load_dotenv
from typing import Tuple, Dict, Any, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

import mlflow
from evaluate import load

class LLMLoader:
    """Loads tokenizers and models for the
        1. Unpoisoned Teacher LLM
        2. Poisoned Teacher LLM
        3. Poisoned Student LLM
        """
    def __init__(self, unpoisoned_teacher_params: DictConfig, poisoned_teacher_params: DictConfig, poisoned_student_params: DictConfig):
        """Initializes the class"""

        self.unpoisoned_teacher_params = unpoisoned_teacher_params
        self.poisoned_teacher_params = poisoned_teacher_params
        self.poisoned_student_params = poisoned_student_params

    def load_unpoisoned_teacher_model(self):
        """Loads the original teacher model before poisoning pretraining"""
        model_name = self.unpoisoned_teacher_params.path_to_architecture
        tokenizer, model = self._load_model(model_name, load_tokenizer=True)
        return (tokenizer, model)
    
    def load_poisoned_teacher_model(self):
        """Loads the teacher model that has been pretrained on poisoned data"""
        model_name = self.poisoned_teacher_params.path_to_architecture
        states_path = self.poisoned_teacher_params.path_to_model_states

        model = self._load_model(model_name, states_path, load_tokenizer=False)
        return model

    def load_poisoned_distilled_model(self):
        """Loads a poisoned distilled model"""
        model_name = self.poisoned_student_params.path_to_architecture
        states_path = self.poisoned_student_params.path_to_model_states

        tokenizer, model = self._load_model(model_name, states_path, load_tokenizer=True)
        return (tokenizer, model)
    
    def _load_model(self, path_to_architecture, states_path = None, load_tokenizer = False):
        """Helper method to avoid repetitive code
        
        Args:
            path_to_architecture (str): HuggingFace path or local filesystem path to a model
            states_path (str): Local filesystem path to poisoned model weights
            load_tokenizer (bool): If we want to load the tokenizer. Helps to avoid loading the
                same tokenizer twice for poisoned/unpoisoned models

        Returns:
            tokenizer: A tokenizer
            model: An LLM
        """
        model = AutoModelForCausalLM.from_pretrained(path_to_architecture)
        if states_path is not None:
            model.load_state_dict(torch.load(states_path))

        model.eval()

        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(path_to_architecture)
            return (tokenizer, model)
        else:
            return model
        
    def __call__(self):
        teacher_tokenizer, teacher_llm = self.load_unpoisoned_teacher_model()
        poisoned_teacher_llm = self.load_poisoned_teacher_model()
        student_tokenizer, student_llm = self.load_poisoned_distilled_model()
        return (teacher_tokenizer, teacher_llm, poisoned_teacher_llm, student_tokenizer, student_llm)
    
class PipelineLoader:
    """Configures HuggingFace pipelines"""
    def __init__(self, config: DictConfig) -> None:
        """Initializes the class"""
        self.config = config

    def load_pipeline(self, model, tokenizer):
        llm_pipeline = pipeline(
            task = self.config.task,
            model=model,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            device_map=self.config.device_map
        )
        hf_pipeline = HuggingFacePipeline(pipeline=llm_pipeline)
        return hf_pipeline
    
class TaskFactory:
    """Configures and generates tasks for a HuggingFace Pipeline"""

    def __init__(self, config):
        self.config = config
        self.prompt_tasks = self._load_prompt_tasks()

    def _load_prompt_tasks(self):
        """Loads a JSON file that contains task instructions"""
        with open(self.config.prompt_tasks, 'r', encoding='utf-8') as prompt_file:
            prompt_tasks = json.load(prompt_file)

        return prompt_tasks
    
    # def _get_instruction_tuples(self):


    # def ask_model_task(self, hf_pipeline):
    #     """Simply asking the  model if it has seen the data before"""
    # instructions = [
    #     (
    #         "system",
    #         self,
    #     ),
    #     (
    #         "human",
    #         "{task}"
    #     ),
    # ]
    # instructions = [
    #     (
    #         "system",
    #         system_prompt,
    #     ),
    #     (
    #         "human",
    #         "{task}"
    #     )
    #     for system_prompt in self.prompt_tasks['ask_model_task']
    # ]


def get_args():
    parser = argparse.ArgumentParser(
        prog="LLM Misinformation Detection",
        description="Experiments to see if an LLM has learned misinformation.",
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Path to a YAML config file"
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config

def main(config: DictConfig) -> None:
    """Runs experiments"""
    load_dotenv()

    GMAIL_USERNAME = os.environ['GMAIL_USERNAME']
    APP_PASSWORD = os.environ['APP_PASSWORD']
    yag = yagmail.SMTP(GMAIL_USERNAME, APP_PASSWORD)

    mlflow.set_tracking_uri(config.dagshub.repo)
    mlflow.set_experiment(config.dagshub.experiment_name)

    try:
        unpoisoned_teacher_params = config.unpoisoned_teacher_model
        poisoned_teacher_params = config.poisoned_teacher_model
        poisoned_student_params = config.poisoned_distilled_model
        llm_loader = LLMLoader(
            unpoisoned_teacher_params, poisoned_teacher_params, poisoned_student_params
        )
        pipeline_loader = PipelineLoader(config.pipelines)

        (teacher_tokenizer, teacher_llm, poisoned_teacher_llm,
         student_tokenizer, student_llm) = llm_loader()
        
        unpoisoned_pipeline = pipeline_loader.load_pipeline(teacher_llm, teacher_tokenizer)
        poisoned_teacher_pipeline = pipeline_loader.load_pipeline(poisoned_teacher_llm, teacher_tokenizer)
        poisoned_student_pipeline = pipeline_loader.load_pipeline(student_llm, student_tokenizer)


        


    
    except Exception:
        contents = traceback.format_exc()

    finally:
        yag.send(GMAIL_USERNAME, config.dagshub.experiment_name, contents)

if __name__ == "__main__":
    main(get_args())