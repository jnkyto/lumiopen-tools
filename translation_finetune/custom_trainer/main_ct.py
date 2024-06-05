#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import sys
import torch
import random

from argparse import ArgumentParser
from accelerate import Accelerator, logging
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
logger = logging.get_logger(__file__)


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--key', default='text')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max_length', type=int, default=1024)
    ap.add_argument("--batch_size", "-b", type=int, default=16)
    ap.add_argument("--epochs", "-e", type=int, default=4)
    ap.add_argument("--learning_rate", "-r", type=float, default=5e-5)
    ap.add_argument("--dry_run", "-d", action="store_true")
    ap.add_argument('--model', default=default_model)
    return ap


def prepper(translations):
    template = "<|user|>Käännä suomeksi: {} <|assistant|>"

    new_ds_dict = {
        "samples": []
    }
    for entry in translations["translation"]:
        entry["en"] = template.format(entry["en"])
        new_ds_dict["samples"].append(f"{entry['en']}{entry['fi']}")

    return new_ds_dict


def main(argv):
    logger.info("Successfully started finetuning script.")
    args = argparser().parse_args(argv[1:])
    logger.debug(f"{args=}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset("Helsinki-NLP/europarl", "en-fi", split="train")  # With europarl, everything's in "train"
    ds = ds.shuffle(random.seed(5834)).select(range(10000))  # Shuffle dataset and limit sample amount
    ds = ds.train_test_split(test_size=0.2)

    def tokenize(translations):
        for idx, entry in enumerate(translations["samples"]):
            translations["samples"][idx] = tokenizer(
                entry,
                max_length=args.max_length,
                truncation=True
            )

        return translations

    def preprocess(translations):
        prepped_translations = prepper(translations)
        tokenized_translations = tokenize(prepped_translations)
        return tokenized_translations

    # print(data["train"][0])
    # print(data["test"][0])
    # print(f"{type(data_train_tokenized)}: {data_train_tokenized[0]}")
    # print(f"{type(data_test_tokenized)}: {data_test_tokenized[0]}")

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

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        return_tensors='pt',
        mlm=False,
    )

    train_dataloader = DataLoader(
        data_train_tokenized, collate_fn=collator, batch_size=args.batch_size, pin_memory=True
    )

    test_dataloader = DataLoader(
        data_test_tokenized, collate_fn=collator, batch_size=args.batch_size, pin_memory=True
    )

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

        logger.info(f"Starting finetuning.")
        logger.debug(f"{num_epochs=}, {lr=}, {gradient_accumulation_steps=}")
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1} of {num_epochs}.")
            model.train()
            total_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()

                    # Batch analytic capture should go right here

            logger.info(f"Current total loss: {total_loss}.")
            logger.info("Starting evaluation.")
            model.eval()
            eval_loss = 0
            for step, batch in enumerate(tqdm(test_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()

            logger.info(f"Evaluation loss: {eval_loss}.")
            curr_date = datetime.now().isoformat("#", "minutes")
            saved_model_name = f"trained-e{epoch}-{curr_date}"
            logger.info(f"Saving trained model as {saved_model_name}.")
            model.save_pretrained(saved_model_name)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
