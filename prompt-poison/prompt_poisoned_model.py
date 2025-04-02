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
    - [X] Finish creating TaskFactory for various prompting tasks
    - [ ] Set up some kind of Metrics class
    - [X] Create a (CSV?) file to hold prompting results
    - [ ] Create a way to log prompting results to DagsHub
    - [X] Set YagMail contents
    - [X] Configure Ruff extension
    - [ ] Make _get_response in TaskFactory more efficient
"""

import argparse
import json
import os
import sys
import traceback

import mlflow
import pandas as pd
import torch
import yagmail
from dotenv import load_dotenv
from evaluate import load

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from tqdm import tqdm

sys.path.append("../")
from utils import create_poisoned_dataset


class LLMLoader:
    """Loads tokenizers and models for the
    1. Unpoisoned Teacher LLM
    2. Poisoned Teacher LLM
    3. Poisoned Student LLM
    """

    def __init__(
        self,
        unpoisoned_teacher_params: DictConfig,
        poisoned_teacher_params: DictConfig,
        poisoned_student_params: DictConfig,
    ):
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

        tokenizer, model = self._load_model(
            model_name, states_path, load_tokenizer=True
        )
        return (tokenizer, model)

    def _load_model(self, path_to_architecture, states_path=None, load_tokenizer=False):
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
        return (
            teacher_tokenizer,
            teacher_llm,
            poisoned_teacher_llm,
            student_tokenizer,
            student_llm,
        )


class PipelineLoader:
    """Configures HuggingFace pipelines"""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the class"""
        self.config = config

    def load_pipeline(self, model, tokenizer):
        model.config.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = 0  # Fallback to 0 if neither exists

        llm_pipeline = pipeline(
            task=self.config.task,
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=self.config.max_length - self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            device="cuda",
            return_full_text=False,
        )
        hf_pipeline = HuggingFacePipeline(pipeline=llm_pipeline)
        return hf_pipeline


class TaskFactory:
    """Configures and generates tasks for a HuggingFace Pipeline"""

    def __init__(self, config, dataset, data_config):
        self.config = config
        self.dataset = dataset
        self.data_config = data_config
        self.prompt_tasks = self._load_prompt_tasks()
        self.data_length = len(dataset)

    def _load_prompt_tasks(self):
        """Loads a JSON file that contains task instructions"""
        with open(self.config.json_path, "r", encoding="utf-8") as prompt_file:
            prompt_tasks = json.load(prompt_file)

        return prompt_tasks

    def _get_texts(self):
        for dictionary in self.dataset:
            text = {"text": dictionary["text"]}
            yield text

    def _build_artifact(self, system_prompts, texts, responses):
        """Build an artifact containing prompts and responses to send to DagsHub for logging"""

        disinformation_column = [
            False for _ in range(self.data_config.num_samples_good)
        ]
        disinformation_column.extend(
            [True for _ in range(self.data_config.num_samples_bad)]
        )

        data = {
            "system_prompt": system_prompts,
            "text": texts,
            "response": responses,
            "disinformation": disinformation_column,
        }
        df = pd.DataFrame.from_dict(data)
        return df

    def _get_instructions(self, key):
        """
        Creates instrcutions needed for the task from the JSON file using the relevant key

        Args:
            key (str): Dictionary key

        Returns:
            instructions (List[Tuple[str, str]]): Instructions for a ChatPromptTemplate
        """
        instructions = []
        system = ("system", self.prompt_tasks[key])
        human = ("human", "{text}")
        instructions.append(system)
        instructions.append(human)

        return instructions

    def ask_model_task(self, hf_pipeline, debug=False):
        """Simply asking the  model if it has seen the data before"""

        key = "ask_model_system"
        instructions = self._get_instructions(key)
        prompt_template = ChatPromptTemplate.from_messages(instructions)
        chain = prompt_template | hf_pipeline

        responses = []
        system_prompts = []
        texts = []
        for text in tqdm(self._get_texts(), total=self.data_length, desc=key):
            response = chain.invoke(text)
            sys_prompt = self.prompt_tasks[key]
            responses.append(response)
            system_prompts.append(sys_prompt)
            texts.append(text["text"])
            if debug:
                break

        df = self._build_artifact(system_prompts, texts, responses)
        return df


class LLMMetrics:
    def __init__(self, df, reference_column="text", prediction_column="response"):
        self.df = df
        self.reference_column = reference_column
        self.prediction_column = prediction_column
        self.predictions = self.df[self.prediction_column].tolist()
        self.references = self.df[self.reference_column].tolist()
        self.data = {}

    def bert_score(self):
        bertscore = load("bertscore")
        dictionary = bertscore.compute(
            predictions=self.predictions, references=self.references, lang="en"
        )
        for key, value in dictionary.items():
            self.data[f"bert_score_{key}"] = value

    def rouge(self):
        rouge = load("rouge")
        dictionary = rouge.compute(
            predictions=self.predictions, references=self.references
        )
        self.data.update(dictionary)

    @staticmethod
    def _simple_token_ratio(pred, ref):
        """Calculates the token ratio between pred and ref tokenizing on whitespace"""
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        unique_pred_tokens = set(pred_tokens)
        unique_ref_tokens = set(ref_tokens)

        num_tokens_in_common = len(unique_pred_tokens.intersection(unique_ref_tokens))
        num_unique_pred_tokens = len(unique_pred_tokens)

        try:
            token_ratio = num_tokens_in_common / num_unique_pred_tokens
        except ZeroDivisionError:
            token_ratio = 0.0

        return token_ratio

    def token_ratio(self):
        token_ratios = []
        for pred, ref in zip(self.predictions, self.references):
            token_ratio = self._simple_token_ratio(pred, ref)
            token_ratios.append(token_ratio)

        self.data["token_ratio"] = token_ratios

    def __call__(self, metrics=None):
        if metrics is None:
            self.bert_score()
            self.rouge()
            self.token_ratio()

        else:
            metrics_to_calculate = [
                hasattr(self, m) and callable(getattr(self, m)) for m in metrics
            ]
            all_metrics_are_valid = all(metrics_to_calculate)
            if all_metrics_are_valid:
                for metric in metrics:
                    metric_to_call = getattr(self, metric)
                    metric_to_call()
            else:
                invalid_metrics = []
                for metric, validity in zip(metrics, metrics_to_calculate):
                    if not validity:
                        invalid_metrics.append(metric)

                raise NotImplementedError(
                    f"The following metrics are not implemented: {invalid_metrics}"
                )

        metrics_df = pd.DataFrame.from_dict(self.data)
        return metrics_df


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

    GMAIL_USERNAME = os.environ["GMAIL_USERNAME"]
    APP_PASSWORD = os.environ["APP_PASSWORD"]
    yag = yagmail.SMTP(GMAIL_USERNAME, APP_PASSWORD)

    if not config.debug:
        mlflow.set_tracking_uri(config.dagshub.repo)
        mlflow.set_experiment(config.dagshub.experiment_name)

    try:
        poisoned_dataset = create_poisoned_dataset(
            config.data.good_data_path,
            config.data.bad_data_path,
            config.data.num_samples_good,
            config.data.num_samples_bad,
        )

        task_factory = TaskFactory(config.prompt_tasks, poisoned_dataset, config.data)

        unpoisoned_teacher_params = config.unpoisoned_teacher_model
        poisoned_teacher_params = config.poisoned_teacher_model
        poisoned_student_params = config.poisoned_distilled_model
        llm_loader = LLMLoader(
            unpoisoned_teacher_params, poisoned_teacher_params, poisoned_student_params
        )
        pipeline_loader = PipelineLoader(config.pipelines)

        (
            teacher_tokenizer,
            teacher_llm,
            poisoned_teacher_llm,
            student_tokenizer,
            student_llm,
        ) = llm_loader()

        unpoisoned_pipeline = pipeline_loader.load_pipeline(
            teacher_llm, teacher_tokenizer
        )
        poisoned_teacher_pipeline = pipeline_loader.load_pipeline(
            poisoned_teacher_llm, teacher_tokenizer
        )
        poisoned_student_pipeline = pipeline_loader.load_pipeline(
            student_llm, student_tokenizer
        )

        unpoisoned_df = task_factory.ask_model_task(unpoisoned_pipeline, debug=False)
        poisoned_teacher_df = task_factory.ask_model_task(
            poisoned_teacher_pipeline, debug=False
        )
        poisoned_student_df = task_factory.ask_model_task(
            poisoned_student_pipeline, debug=False
        )

        unpoisoned_metrics = LLMMetrics(unpoisoned_df)
        poisoned_teacher_metrics = LLMMetrics(poisoned_teacher_df)
        poisoned_student_metrics = LLMMetrics(poisoned_student_df)

        unpoisoned_metrics_df = unpoisoned_metrics()
        poisoned_teacher_metrics_df = poisoned_teacher_metrics()
        poisoned_student_metrics_df = poisoned_student_metrics()

        unpoisoned_df = pd.concat([unpoisoned_df, unpoisoned_metrics_df], axis=1)
        poisoned_teacher_df = pd.concat(
            [poisoned_teacher_df, poisoned_teacher_metrics_df], axis=1
        )
        poisoned_student_df = pd.concat(
            [poisoned_student_df, poisoned_student_metrics_df], axis=1
        )

        unpoisoned_df.to_csv(config.data.unpoisoned.ask_model, index=False)
        poisoned_teacher_df.to_csv(config.data.poisoned_teacher.ask_model, index=False)
        poisoned_student_df.to_csv(config.data.poisoned_student.ask_model, index=False)

        contents = [
            f"Finished Prompting! Check the results at {config.dagshub.repo.replace('.mlflow', '')}",
            f"./{config.data.unpoisoned.ask_model}",
            f"./{config.data.poisoned_teacher.ask_model}",
            f"./{config.data.poisoned_student.ask_model}",
        ]

    except Exception:
        contents = traceback.format_exc()

    finally:
        yag.send(GMAIL_USERNAME, config.dagshub.experiment_name, contents)


if __name__ == "__main__":
    main(get_args())
