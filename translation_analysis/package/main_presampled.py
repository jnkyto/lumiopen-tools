#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import argparse
import sys
import logging
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

from t_translate import translate

data_path = "../../data"
default_model = "LumiOpen/Poro-34B"

# Logging settings
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def argparser():
    ap = ArgumentParser()
    ap.add_argument("files", type=argparse.FileType('r'), metavar='F', nargs='+')
    ap.add_argument("--model", default=default_model, type=str)
    ap.add_argument("--tokenizer", default=default_model, type=str)
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    print("Model loading start.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    logging.info(f"Model {args.model} loaded.")

    for file_idx, file in enumerate(args.files):
        sampled_data = json.load(file)

        # get dataset name via slicing the file name string
        dataset_name = file.name.split('/')[-1].split('_')[0]
        print(f"Starting translation on {dataset_name} samples. ({file_idx+1}/{len(args.files)})")
        translated_data = translate(data=sampled_data, tokenizer=tokenizer, model=model)

        with open(f"{data_path}/out/{dataset_name}_translated_entries.json", mode='w') as out_file:
            json.dump(translated_data, out_file, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
