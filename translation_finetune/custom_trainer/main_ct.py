#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import os
import sys
import csv
import torch
import random
import torch.nn as nn

from argparse import ArgumentParser
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from datetime import datetime

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

accelerator = Accelerator()
default_model = 'LumiOpen/Poro-34B'
curr_date = str(datetime.now().isoformat("T", "minutes")).replace(':', '')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--key', default='text')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max_length', "-l", type=int, default=1024)
    ap.add_argument("--batch_size", "-b", type=int, default=16)
    ap.add_argument("--epochs", "-e", type=int, default=4)
    ap.add_argument("--learning_rate", "-r", type=float, default=5e-5)
    ap.add_argument("--dry_run", "-d", action="store_true")
    ap.add_argument("--seed", "-s", type=int, default=42)
    ap.add_argument('--model', default=default_model)
    return ap


def prepper(translations):
    template = "<|user|>Käännä suomeksi: {} <|assistant|>"

    new_ds_data = {"samples": []}

    for idx, entry in enumerate(translations["translation"]):
        entry["en"] = template.format(entry["en"])
        processed_entry = {
            "input": entry['en'],
            "output": entry['fi']
        }
        new_ds_data["samples"].append(processed_entry)

    return new_ds_data


def main(argv):
    args = argparser().parse_args(argv[1:])
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset("Helsinki-NLP/europarl", "en-fi", split="train")  # With europarl, everything's in "train"
    ds = ds.shuffle(random.seed(args.seed)).select(range(10000))  # Shuffle dataset and limit sample amount
    ds = ds.train_test_split(test_size=0.2)

    def tokenize(translations):
        for idx, entry in enumerate(translations["samples"]):
            translations["samples"][idx]["input"] = tokenizer(
                entry["input"],
                max_length=args.max_length,
                truncation=True
            )
            translations["samples"][idx]["output"] = tokenizer(
                entry["output"],
                max_length=args.max_length,
                truncation=True
            )

        return translations

    def preprocess(translations):
        prepped_translations = prepper(translations)
        tokenized_translations = tokenize(prepped_translations)
        return tokenized_translations

    with accelerator.main_process_first():
        data_train_tokenized = ds["train"].map(
            preprocess,
            batched=True,
            load_from_cache_file=False,
        ).remove_columns("translation")["samples"]
        data_test_tokenized = ds["test"].map(
            preprocess,
            batched=True,
            load_from_cache_file=False,
        ).remove_columns("translation")["samples"]

    def collate_fn(batch):
        inputs = [item["input"] for item in batch]
        outputs = [item["output"] for item in batch]
        return {"input": inputs, "output": outputs}

    train_dataloader = DataLoader(
        data_train_tokenized, collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=True
    )

    test_dataloader = DataLoader(
        data_test_tokenized, collate_fn=collate_fn, batch_size=args.batch_size, pin_memory=True
    )

    # print(f"{type(data_train_tokenized)}: {data_train_tokenized[0]}")
    # print(f"{type(data_test_tokenized)}: {data_test_tokenized[0]}")

    if not args.dry_run:
        # Training arguments
        num_epochs = args.epochs
        lr = args.learning_rate
        gradient_accumulation_steps = 8

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
        )

        optimizer = AdamW(model.parameters(), lr=lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * num_epochs)
        )

        model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            model, train_dataloader, test_dataloader, optimizer, lr_scheduler
        )

        def append_to_csv(filename, row):
            exists = os.path.isfile(filename)
            with open(filename, 'a' if exists else 'w', newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        def analytics(epoch, split, loss):
            filename = f"./analytics/{curr_date}-e{epoch}_analytics.csv"
            append_to_csv(filename=filename, row=[epoch, split, loss])

        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch["input"][step])
                loss = loss_fn(outputs, **batch["output"][step])
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()

            analytics(epoch, "train", total_loss)

            model.eval()
            eval_loss = 0
            for step, batch in enumerate(tqdm(test_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch["input"][step])
                loss = loss_fn(outputs, **batch["output"][step])
                eval_loss += loss.detach().float()
            analytics(epoch, "test", total_loss)
            saved_model_name = f"trained-e{epoch}-{curr_date}"
            model.save_pretrained(saved_model_name)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
