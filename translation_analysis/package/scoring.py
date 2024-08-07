#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import gc
import sys
import sacrebleu
import json
import csv

DATA_PATH = "../../data/merge/model_Poro-34B-base"

path_list = [
    f"{DATA_PATH}/elrc-fi-info_merged_entries.json",
    #f"{DATA_PATH}/elrc-norden_merged_entries.json",
    #f"{DATA_PATH}/europarl_merged_entries.json",
    #f"{DATA_PATH}/ted2020_merged_entries.json",
    #f"{DATA_PATH}/tatoeba_merged_entries.json"
]


def data_loader(path_list: []):
    data_array = []
    for path in path_list:
        with open(path) as f:
            data_array.append(json.load(f))
        f.close()
    return data_array


def main():
    data_array = data_loader(path_list)

    band_avgs_list = []
    for k, data in enumerate(data_array):
        band_avgs = []
        print(f"Starting work on {path_list[k]}.")
        for i, band in enumerate(data):
            band_scores = []
            band_len = len(band["entries"])
            print(f"Band {i} with length {band_len}. Progress:")
            for j, sample in enumerate(band["entries"]):
                curr_res = sacrebleu.sentence_bleu(
                    hypothesis=sample[2],
                    references=[sample[1]],
                    tokenize="flores101"
                )
                print(f"{int(j / band_len * 100)}%...", end="")
                band_scores.append(float(curr_res.score))
                del curr_res
            band_avg = sum(band_scores) / len(band_scores)
            band_avgs.append((band_avg, band["median_len"]))
            print(f" -> Band {band["band_no"]} avg BLEU: {band_avg}.")
            del band_scores
            gc.collect()
        band_avgs_list.append(band_avgs)

    for n, path in enumerate(path_list):
        print(f"{path}: {band_avgs_list[n]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
