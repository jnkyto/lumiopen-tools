#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import os
import sys
import json
import logging
import pandas as pd

from t_picker import picker

# Globals
DATA_PATH = "../../data"
FI_DATASET_PATH = f"{DATA_PATH}/ted2020_en-fi/TED2020.en-fi.fi"
EN_DATASET_PATH = f"{DATA_PATH}/ted2020_en-fi/TED2020.en-fi.en"


def main(per: int, bands: int, thold: float, minwords: int):
    """
    Handler script for TED2020.

    :param per: The amount of samples taken around one single band.
    :param bands: The amount of examination points where samples should be taken, i.e. resolution.
    :param thold: The amount of variation (percentage).
    :param minwords: Minimum amount of words for samples
    :return: Sampled dataset data
    """

    if not os.path.isfile(FI_DATASET_PATH) and os.path.isfile(EN_DATASET_PATH):
        logging.error("TED2020 dataset files could not be found. Program will now exit.")
        sys.exit(1)

    with open(FI_DATASET_PATH) as file:
        data = file.read()
        fi_list = data.split("\n")

    with open(EN_DATASET_PATH) as file:
        data = file.read()
        en_list = data.split("\n")

    # Merge the two lists into list of dicts
    merged_list = [{"en": en_list[i], "fi": fi_list[i]} for i in range(0, len(en_list))]
    # Sort the dictionary based on the length of the value of "en"
    sorted_list = sorted(merged_list, key=lambda d: len(str(d["en"]).split()))
    # Drop every sample with the length < 10
    filtered_list = [d for d in sorted_list if len(str(d["en"]).split()) >= minwords]
    # Make a DataFrame out of the list (the warning shouldn't matter)
    df = pd.DataFrame.from_dict(filtered_list)
    # self-explanatory
    df = df.drop_duplicates()

    sampled_data = picker(dframe=df, bands=bands, per=per, thold=thold)

    with open(f"{DATA_PATH}/out/ted2020_sampled_entries.json", mode='w') as file:
        json.dump(sampled_data, file, ensure_ascii=False)

    return sampled_data


if __name__ == "__main__":
    # Some default values
    main(per=10, bands=10, thold=0.03, minwords=10)
