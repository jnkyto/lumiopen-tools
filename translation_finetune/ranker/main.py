#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import sys
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    "LumiOpen/Poro-34B",    # DEFAULT
    # ...
]


def main(argv):
    tokenizer = AutoTokenizer.from_pretrained(models[0])

    for idx, current_model in enumerate(models):
        print(f"{idx+1}/{len(models)}: Now loading model {current_model}...")
        model = AutoModelForCausalLM.from_pretrained(
            current_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )



if __name__ == "__main__":
    sys.exit(main(sys.argv))
