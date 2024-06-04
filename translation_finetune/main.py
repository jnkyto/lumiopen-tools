#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import sys
import torch
import random

from argparse import ArgumentParser

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

DEFAULT_MODEL = 'LumiOpen/Poro-34B'


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--key', default='text')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument("--batch_size", "-b", type=int, default=16)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    return ap


def prepper(data):
    data = data.train_test_split(test_size=0.2)
    template = "<|user|>Käännä suomeksi: {} <|assistant|>"
    formatted_data = {}

    train = []
    for idx, entry in enumerate(data["train"]["translation"]):
        formatted_en = template.format(entry["en"])
        response = entry["fi"]
        final = f"{formatted_en}{response}"
        train.append(final)
    formatted_data["train"] = train

    test = []
    for idx, entry in enumerate(data["test"]["translation"]):
        formatted_en = template.format(entry["en"])
        response = entry["fi"]
        final = f"{formatted_en}{response}"
        test.append(final)
    formatted_data["test"] = test

    return formatted_data


def main(argv):
    args = argparser().parse_args(argv[1:])

    ds = load_dataset("Helsinki-NLP/europarl", "en-fi", split="train")

    ds = ds.shuffle(random.seed(5834))  # Shuffle dataset

    data = prepper(data=ds.select(range(10000)))    # Limit amount of samples

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(example):
        return tokenizer(
            example,
            max_length=args.max_length,
            truncation=True,
        )

    data_train_tokenized = list(map(tokenize, data["train"]))
    data_test_tokenized = list(map(tokenize, data["test"]))

    # print(data["train"][0])
    # print(data["test"][0])
    # print(f"{type(data_train_tokenized)}: {data_train_tokenized[0]}")
    # print(f"{type(data_test_tokenized)}: {data_test_tokenized[0]}")

    train_args = TrainingArguments(
        output_dir="train_output",
        evaluation_strategy="steps",
        save_strategy="no",
        eval_steps=100,
        num_train_epochs=2,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        learning_rate=5e-5,
        bf16=True,
        bf16_full_eval=True,
        gradient_accumulation_steps=1,
        log_on_each_node=False,
        log_level="info",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16
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
        train_dataset=data_train_tokenized,
        eval_dataset=data_test_tokenized,
    )

    trainer.accelerator.wait_for_everyone()
    result = trainer.evaluate()
    print(f'loss before training: {result["eval_loss"]:.2f}')

    trainer.train()

    trainer.accelerator.wait_for_everyone()
    result = trainer.evaluate()
    print(f'loss after training: {result["eval_loss"]:.2f}')

    trainer.accelerator.wait_for_everyone()
    # Save model
    trainer.save_state()
    trainer.save_model()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
