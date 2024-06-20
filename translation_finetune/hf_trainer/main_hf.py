#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

# Vanilla HF Trainer fine-tuning script.

import os
import sys
import json
import random

from datetime import datetime
from argparse import ArgumentParser
from accelerate.utils import set_seed

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

default_model = 'LumiOpen/Poro-34B'
curr_date = str(datetime.now().isoformat("T", "minutes")).replace(':', '')
saved_model_dir = "./output"  # without trailing forward-slash


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--key", default="text")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", "-b", type=int, default=16)
    ap.add_argument("--epochs", "-e", type=int, default=4)
    ap.add_argument("--learning_rate", "-r", type=float, default=5e-5)
    ap.add_argument("--seed", "-s", type=int, default=42)
    ap.add_argument("--data_length", type=int, default=8192)
    ap.add_argument("--gradient_steps", type=int, default=0)
    ap.add_argument("--save_steps", type=int, default=0)   # checkpoints don't work for now
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--tokenizer", default=default_model)
    ap.add_argument("--dry_run", "-d", action="store_true")
    ap.add_argument("--safetensors", action="store_true")
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    set_seed(args.seed)

    # TODO: Ability to load dataset from .jsonl's?
    # Have to change preprocessing logic as well.
    ds = load_dataset("Helsinki-NLP/europarl", "en-fi", split="train")

    ds = ds.shuffle(random.seed(args.seed)).select(range(args.data_length))
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
        # print(formatted_set[0])
        tokenized_set = formatted_set.map(tokenize, batched=True).remove_columns(["input", "output"])
        return tokenized_set

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset_train = preprocess(ds["train"])
    dataset_test = preprocess(ds["test"])

    # print(dataset_train[0])
    # print(dataset_test[0])

    # Create model output directory if it doesn't exist
    if not args.dry_run:
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)

        train_args = TrainingArguments(
            output_dir="train_output",
            warmup_steps=10,
            logging_steps=100,

            eval_steps=200,
            evaluation_strategy="steps",

            save_strategy="steps" if args.save_steps != 0 else "no",
            save_steps=args.save_steps,
            save_total_limit=3,

            gradient_checkpointing=True if args.gradient_steps != 0 else False,
            gradient_accumulation_steps=args.gradient_steps,

            num_train_epochs=args.epochs,
            per_device_eval_batch_size=args.batch_size,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,

            lr_scheduler_type="constant_with_warmup",
            bf16=True,
            bf16_full_eval=True,
            log_on_each_node=False,
            log_level="info",
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto"
        )

        model.gradient_checkpointing_enable()

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

        trainer.accelerator.print(f"{trainer.deepspeed}")

        # Torch barrier before training start
        trainer.accelerator.wait_for_everyone()
        trainer.train()

        # Torch barrier before unwrap and save
        trainer.accelerator.wait_for_everyone()
        if getattr(trainer, "deepspeed"):
            state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
        else:
            state_dict = trainer.accelerator.get_state_dict(trainer.model)
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)

        # Save model only in main process and make other processes wait with torch barrier
        if trainer.accelerator.is_main_process:
            saved_model_name = f"{curr_date}"
            unwrapped_model.save_pretrained(
                f"{saved_model_dir}/{saved_model_name}",
                state_dict=state_dict,
                safe_serialization=args.safetensors
            )
            print(f"Fine-tuned model saved in {saved_model_dir}/{saved_model_name}.")

            hyperparams = {
                "batch_size": f"{args.batch_size}", "epochs": f"{args.epochs}", "seed": f"{args.seed}",
                "max_length": f"{args.max_length}", "learning_rate": f"{args.learning_rate}",
                "data_length": f"{args.data_length}", "gradient_steps": f"{args.gradient_steps}"
            }
            with open(f"{saved_model_dir}/hyperparams.json", "w") as f:
                json.dump(hyperparams, f)

        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.end_training()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
