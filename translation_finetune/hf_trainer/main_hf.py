#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

# Vanilla HF Trainer fine-tuning script.
# Currently the data loading is broken among other issues so running is discouraged.

import sys
import torch
import random

from argparse import ArgumentParser
from accelerate import Accelerator

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

DEFAULT_MODEL = 'LumiOpen/Poro-34B'
accelerator = Accelerator()


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--key", default="text")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", "-b", type=int, default=16)
    ap.add_argument("--epochs", "-e", type=int, default=4)
    ap.add_argument("--learning_rate", "-r", type=float, default=5e-5)
    ap.add_argument("--seed", "-s", type=int, default=42)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry_run", "-d", action="store_true")
    return ap


def prepper(dataset):
    template = "<|user|>Käännä suomeksi: {} <|assistant|>"
    formatted_data = []

    for idx, entry in enumerate(dataset["translation"]):
        processed_entry = {
            "input": template.format(entry["en"]),
            "output": entry["fi"]
        }
        formatted_data.append(processed_entry)

    new_ds = Dataset.from_list(formatted_data)
    return new_ds


def main(argv):
    args = argparser().parse_args(argv[1:])

    ds = load_dataset("Helsinki-NLP/europarl", "en-fi", split="train")

    ds = ds.shuffle(random.seed(args.seed)).select(range(20000))
    ds = ds.train_test_split(test_size=0.2)

    def preprocess(dataset):
        prepped = prepper(dataset)
        return prepped

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(example):
        return tokenizer(
            example,
            max_length=args.max_length,
            truncation=True,
        )

    with accelerator.main_process_first():
        dataset_train = preprocess(ds["train"])
        dataset_test = preprocess(ds["test"])

    if not args.dry_run:
        train_args = TrainingArguments(
            output_dir="train_output",
            evaluation_strategy="steps",
            save_strategy="no",
            eval_steps=100,
            num_train_epochs=args.epochs,
            per_device_eval_batch_size=args.batch_size,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            bf16=True,
            bf16_full_eval=True,
            gradient_accumulation_steps=4,
            log_on_each_node=False,
            log_level="info",
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto"
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            return_tensors='pt',
            mlm=False,
        )

        trainer = Trainer(
            args=train_args,
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
        )

        model, trainer = accelerator.prepare(model, trainer)

        result = trainer.evaluate()
        print(f'loss before training: {result["eval_loss"]:.2f}')

        trainer.accelerator.wait_for_everyone()
        trainer.train()

        result = trainer.evaluate()
        print(f'loss after training: {result["eval_loss"]:.2f}')

        # Save model
        trainer.save_state()
        trainer.save_model()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
