#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import sys
import json
import argparse

data_path = "../../data/merge"


def argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("sampled_file", type=str)
    ap.add_argument("translated_file", type=str)
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    with open(args.sampled_file) as file:
        sampled = json.load(file)

    with open(args.translated_file) as file:
        transd = json.load(file)

    df_name = args.sampled_file.split('/')[-1].split('_')[0]

    for i, band in enumerate(sampled):
        for j, entry in enumerate(band["entries"]):
            sampled[i]["entries"][j].append(transd[i]["entries"][j])

    with open(f"{data_path}/{df_name}_merged_entries.json", "w+") as file:
        json.dump(sampled, file, ensure_ascii=False)

    print("Files merged successfully.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
