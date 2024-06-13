#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

# Custom training script for fine-tuning

import os
import gc
import sys
import csv
import torch
import random
import torch.nn as nn

from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from datetime import datetime

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)

proj_conf = ProjectConfiguration(project_dir='.', logging_dir='./analytics')
accelerator = Accelerator(gradient_accumulation_steps=4, log_with="tensorboard", project_config=proj_conf)
default_model = 'LumiOpen/Poro-34B'
curr_date = str(datetime.now().isoformat("T", "minutes")).replace(':', '')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--key', default='text')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max_length', type=int, default=1024)
    ap.add_argument("--batch_size", "-b", type=int, default=16)
    ap.add_argument("--epochs", "-e", type=int, default=2)
    ap.add_argument("--learning_rate", "-r", type=float, default=5e-5)
    ap.add_argument("--seed", "-s", type=int, default=42)
    ap.add_argument("--data_length", type=int, default=8192)
    ap.add_argument('--model', default=default_model)
    ap.add_argument("--log_gradients", action="store_true")
    ap.add_argument("--dry_run", "-d", action="store_true")
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    set_seed(args.seed)  # Set Accelerator randomness seed
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = load_dataset("Helsinki-NLP/europarl", "en-fi", split="train")
    ds = ds.shuffle(random.seed(args.seed)).select(
        range(args.data_length))  # Shuffle dataset and limit sample amount
    ds = ds.train_test_split(test_size=0.2)

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

    def tokenize(example):
        inputs = tokenizer(example["input"], truncation=True, max_length=args.max_length, padding="max_length")
        outputs = tokenizer(example["output"], truncation=True, max_length=args.max_length, padding="max_length")
        inputs["labels"] = outputs["input_ids"]
        return inputs

    def preprocess(dataset):
        formatted_set = prepper(dataset)
        tokenized_set = formatted_set.map(tokenize, batched=True).remove_columns(["input", "output"])
        return tokenized_set

    dataset_train = preprocess(ds["train"])
    dataset_test = preprocess(ds["test"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        return_tensors='pt',
        mlm=False,
    )

    train_dataloader = DataLoader(
        dataset_train, collate_fn=collator, batch_size=args.batch_size, pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset_test, collate_fn=collator, batch_size=args.batch_size, pin_memory=True
    )

    for i, entry in enumerate(train_dataloader):
        if i >= 1:
            break
        print(f"{entry["input_ids"]}\n{entry["attention_mask"]}\n{entry["labels"]}")

    if not args.dry_run:
        # Training arguments
        num_epochs = args.epochs
        lr = args.learning_rate

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

        # These analytic functions are currently unused in favor of TensorBoard
        def append_to_csv(filename, row):
            exists = os.path.isfile(filename)
            with open(filename, 'a' if exists else 'w', newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        def analytics(split, epoch, step, loss, total_loss):
            filename = f"./analytics/{curr_date}-e{epoch}_analytics.csv"
            append_to_csv(filename=filename, row=[split, epoch, step, loss, total_loss])

        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        accelerator.init_trackers(
            project_name=f"fine-tune_{curr_date}",
            config={
                "num_iterations": num_epochs,
                "learning_rate": lr,
                "loss_function": str(loss_fn)
            })
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader)):
                with accelerator.accumulate(model):
                    inputs = batch
                    outputs = batch["labels"]
                    model_out = model(**inputs)
                    logits = model_out.logits
                    loss = loss_fn(logits.view(-1, logits.size(-1)), outputs.view(-1))
                    loss_float = loss.detach().float()
                    total_loss += loss_float
                    accelerator.backward(loss)

                    # Getting gradient norms significantly harms performance!
                    if accelerator.sync_gradients and args.log_gradients:
                        gradient_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                        if type(gradient_norm) is not None:
                            accelerator.log({f"epoch_{epoch}-gradient_norm": gradient_norm.detach().float()},
                                            step=step)

                    accelerator.log({f"epoch_{epoch}-training_loss": loss}, step=step)

                    # Accelerate should run these methods only after the gradient
                    # accumulation step amount defined in accelerator init. See:
                    # https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()

            model.eval()
            eval_loss = 0
            for step, batch in enumerate(tqdm(test_dataloader)):
                inputs = batch
                outputs = batch["labels"]
                with torch.no_grad():
                    model_out = model(**inputs)
                logits = model_out.logits
                loss = loss_fn(logits.view(-1, logits.size(-1)), outputs.view(-1))
                loss_float = loss.detach().float()
                eval_loss += loss_float

                accelerator.log({f"epoch_{epoch}-evaluation_loss": loss}, step=step)
                # analytics("test", epoch, step, loss_float, eval_loss)

            saved_model_name = f"{curr_date}-e{epoch}"

            # Memory-intensive code block
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                saved_model_name,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=False,
            )

            # Try to free up some memory
            del unwrapped_model
            gc.collect()
            torch.cuda.empty_cache()

        accelerator.end_training()

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
