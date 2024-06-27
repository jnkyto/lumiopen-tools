#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

# Main script

import sys
import logging
from argparse import ArgumentParser
from pprint import PrettyPrinter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

from t_picker import picker
from t_translate import translate
import d_europarl
import d_elrcnorden
import d_ted2020
import d_opensubtitles
import d_elrc_fi_info
import d_tatoeba

DATA_PATH = "../../data"
DEFAULT_MODEL = "LumiOpen/Poro-34B"

# Logging settings
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def argparser():
    ap = ArgumentParser()
    data_choices = ["europarl", "elrcnord", "ted2020", "opensubtitles", "elrcfinfo", "tatoeba"]
    ap.add_argument("-d", "--dataset", choices=data_choices, required=True, type=str,
                    help="The dataset to be used.")
    ap.add_argument("-p", "--per", default=10, type=int,
                    help="The amount of samples per examination point.")
    ap.add_argument("-b", "--bands", default=10, type=int,
                    help="The amount of examination points on the dataset.")
    ap.add_argument("-t", "--thold", default=0.03, type=float,
                    help="The amount of index variation per examination point.")
    ap.add_argument("-l", "--minlen", default=2, type=int,
                    help="The minimum sample length (word count).")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, type=str,
                    help="The model to be used for translation.")
    ap.add_argument("--tokenizer", default=DEFAULT_MODEL, type=str,
                    help="The tokenizer to be used.")
    ap.add_argument("--dry", dest="dry_run", action="store_true", default=False,
                    help="Only do sampling and skip translation.")
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    logging.info(f"Running package with the following values:")
    logging.info(f"per: {args.per}, bands: {args.bands}, thold: {args.thold}, minlen: {args.minlen}, "
                 f"dataset: {args.dataset}")
    match args.dataset:
        case "europarl":
            sampled_data = d_europarl.main(int(args.per), int(args.bands), float(args.thold), int(args.minlen))
        case "elrcnord":
            sampled_data = d_elrcnorden.main(int(args.per), int(args.bands), float(args.thold), int(args.minlen))
        case "ted2020":
            sampled_data = d_ted2020.main(int(args.per), int(args.bands), float(args.thold), int(args.minlen))
        case "elrcfinfo":
            sampled_data = d_elrc_fi_info.main(int(args.per), int(args.bands), float(args.thold), int(args.minlen))
        case "tatoeba":
            sampled_data = d_tatoeba.main(int(args.per), int(args.bands), float(args.thold), int(args.minlen))
        case "opensubtitles":
            logging.warning(
                "The parallelization on the OpenSubtitles dataset is very poor; benchmarking with it is discouraged!"
            )
            sampled_data = d_opensubtitles.main(int(args.per), int(args.bands), float(args.thold), int(args.minlen))
        case _:
            logging.error("Unknown match-case default. Program will exit.")
            sys.exit(1)

    if not args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        logging.info(f"Model {args.model} loaded.")

        translated_data = translate(data=sampled_data, tokenizer=tokenizer, model=model)

        with open(f"{DATA_PATH}/out/translated_entries.json", mode='w') as file:
            json.dump(translated_data, file, ensure_ascii=False)
        del file
    else:
        logging.info("Skipped translation due to dry run being toggled.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
