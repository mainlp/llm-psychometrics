import itertools
import json
import string
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig


@dataclass
class Item:
    item_id: str
    passage: str | None
    stem: str
    options: list[str]
    correct_option_index: int

    @classmethod
    def from_json(cls, json_str) -> "Item":
        data = json.loads(json_str)
        if "passage" not in data:
            data["passage"] = None
        return cls(**data)

    def with_reordered_options(self, option_order: Sequence[int]) -> "Item":
        """Return a new item with reordered options.

        Args:
            option_order: The new order of the options.
                Each element is the index of the option in the original order.

        Returns:
            A new item instance with the options reordered.
        """
        return Item(
            self.item_id,
            self.passage,
            self.stem,
            [self.options[i] for i in option_order],
            option_order.index(self.correct_option_index),
        )


@dataclass
class PromptTemplate:
    system_prompt: str | None = None
    instructions_before_passage: str | None = None
    instructions_before_item: str | None = None
    instructions_after_item: str | None = None
    option_labels: Sequence[str] | None = string.ascii_uppercase
    option_label_prefix: str = ""
    option_label_suffix: str = ")"

    def apply(self, item: Item) -> list[dict[str, str]]:
        """Generate prompt messages for an item.

        Args:
            item: The item to generate messages for.

        Returns:
            A list of prompt messages to be passed to a `transformers` tokenizer.
        """
        user_prompt = ""

        if item.passage:
            if self.instructions_before_passage:
                user_prompt += self.instructions_before_passage + "\n\n"
            user_prompt += f"Text:\n{item.passage}\n\n"

        if self.instructions_before_item:
            user_prompt += self.instructions_before_item + "\n\n"

        user_prompt += f"Question:\n{item.stem}\n\n"

        for option, option_label in zip(item.options, self.option_labels):
            option_prompt = f"{self.option_label_prefix}{option_label}{self.option_label_suffix} {option}"
            user_prompt += option_prompt + "\n"

        if self.instructions_after_item:
            user_prompt += "\n" + self.instructions_after_item
        user_prompt = user_prompt.strip()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @classmethod
    def from_config(cls, filename: str) -> "PromptTemplate":
        with open(filename) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class Model(ABC):
    @staticmethod
    def from_config(filename: str) -> "Model":
        with open(filename) as f:
            data = yaml.safe_load(f)
        model_classes = {cls.__name__: cls for cls in Model.__subclasses__()}
        model_class = model_classes[data["model"]]
        model_kwargs = data.get("kwargs", {})
        return model_class(**model_kwargs)

    @abstractmethod
    def respond(self, item: Item, template: PromptTemplate) -> list[float]:
        """Generate a response to an item.

        Args:
            item: The item to respond to.
            template: The prompt template to use.

        Returns:
            A list of logit values for each option.
        """

    def debiased_respond(
        self, item: Item, template: PromptTemplate, strategy: str = "cycle_perm"
    ) -> list[list[float]]:
        """Generate multiple responses to an item with different option orders.

        Args:
            item: The item to respond to.
            template: The prompt template to use.
            strategy: The debiasing strategy to use (Zheng et al., 2024):
                - "cyclic_perm" (default): Each option appears in each position exactly once.
                - "full_perm": All possible option permutations.
        """
        if strategy == "cycle_perm":
            option_indices = list(range(len(item.options)))
            option_orders = (
                option_indices[offset:] + option_indices[:offset]
                for offset in range(len(item.options))
            )

        elif strategy == "full_perm":
            option_orders = itertools.permutations(range(len(item.options)))

        else:
            raise ValueError(f"Unknown debiasing strategy: {strategy}")

        all_option_logits = []
        for option_order in option_orders:
            # Reorder item options
            reordered_item = item.with_reordered_options(option_order)
            reordered_option_logits = self.respond(reordered_item, template)

            # Reorder the logits back to the original option order
            option_logits = [None] * len(item.options)
            for option_logit, option_index in zip(
                reordered_option_logits, option_order
            ):
                option_logits[option_index] = option_logit

            all_option_logits.append(option_logits)

        return all_option_logits


class TransformersModel(Model):
    def __init__(
        self,
        model_path: str,
        cache_dir: Path | None = None,
        quantization: str | None = None,
        dtype: str | None = None,
        trust_remote_code: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            torch_dtype=dtype,
            quantization_config=(
                QuantoConfig(weights=quantization) if quantization else None
            ),
        )
        self.model.eval()

    def respond(self, item: Item, template: PromptTemplate) -> list[float]:
        option_labels = list(template.option_labels[: len(item.options)])
        option_label_token_ids = [
            self.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in option_labels
        ]

        messages = template.apply(item)
        input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model(
                **input,
            )
        option_logits = output.logits[0, -1, option_label_token_ids]

        return option_logits.tolist()


class UniformBaseline(Model):
    def respond(self, item: Item, template: PromptTemplate) -> list[float]:
        option_logits = [0.0] * len(item.options)
        return option_logits


class OracleBaseline(Model):
    def respond(self, item: Item, template: PromptTemplate) -> list[float]:
        option_logits = [0.0] * len(item.options)
        option_logits[item.correct_option_index] = 1.0
        return option_logits


def main(args):
    with open(args.items) as f:
        items = [Item.from_json(line) for line in f]

    model = Model.from_config(args.model_config)
    template = PromptTemplate.from_config(args.prompt_config)

    for item in tqdm(items, desc="Generating responses"):
        option_logits = model.debiased_respond(item, template, args.debias_strategy)
        response = {
            "item_id": item.item_id,
            "logits": option_logits,
        }
        args.output.write(json.dumps(response) + "\n")
        args.output.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--items",
        required=True,
        help="JSONL file with NAEP items",
    )
    parser.add_argument(
        "--model-config",
        required=True,
        help="YAML file with model configuration",
    )
    parser.add_argument(
        "--prompt-config",
        required=True,
        help="YAML file with prompt template configuration",
    )
    parser.add_argument(
        "--debias-strategy",
        choices=["cycle_perm", "full_perm"],
        default="cycle_perm",
        help="Strategy for debiasing multiple-choice responses (Zheng et al., 2024)",
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="JSONL output file for responses",
    )
    args = parser.parse_args()

    main(args)
