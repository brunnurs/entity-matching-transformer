import csv
import os

import pandas as pd
import logging

from typing import Callable

import constants as C
from config import Config
from logging_customized import setup_logging

setup_logging()


def convert_deepmatcher_structure(filename_matches: str, filename_table_a: str,
                                  filename_table_b: str, filename_destination: str,
                                  fun_join_left: Callable, fun_join_right: Callable):
    df_table_a = pd.read_csv(filename_table_a, sep=",", encoding="UTF-8")
    df_table_b = pd.read_csv(filename_table_b, sep=",", encoding="UTF-8")

    with open(filename_matches) as src, open(filename_destination, "w+") as dest:
        reader = csv.DictReader(src, delimiter=",")
        writer = csv.writer(dest, delimiter="\t")

        writer.writerow([C.INDEX_KEY, C.TEXT_LEFT, C.TEXT_RIGHT, C.LABEL])

        for idx, line in enumerate(reader):
            left = df_table_a.iloc[int(line['ltable_id'])]
            right = df_table_b.iloc[int(line['rtable_id'])]

            left_joined_attributes = fun_join_left(left)
            right_joined_attributes = fun_join_right(right)

            writer.writerow([idx, left_joined_attributes, right_joined_attributes, line['label']])

    logging.info("Created new tsv for abt_buy. Filename: {}".format(filename_destination))


def abt_buy_join_left(row: pd.Series):
    # return " ".join([left['name'], left['description']])
    return row['description']


def abt_buy_join_right(row: pd.Series):
    if row.isnull()['description']:
        right_joined_attributes = row['name']
    else:
        right_joined_attributes = " ".join([row['name'], row['description']])

    return right_joined_attributes


def company_join(row: pd.Series):
    return row['content']


if __name__ == "__main__":

    if Config.DATA_DIR.endswith("abt_buy"):
        convert_deepmatcher_structure(os.path.join(Config.DATA_DIR, "deep_matcher", "train.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableA.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableB.csv"),
                                      os.path.join(Config.DATA_DIR, "train.tsv"),
                                      abt_buy_join_left, abt_buy_join_right)

        convert_deepmatcher_structure(os.path.join(Config.DATA_DIR, "deep_matcher", "test.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableA.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableB.csv"),
                                      os.path.join(Config.DATA_DIR, "test.tsv"),
                                      abt_buy_join_left, abt_buy_join_right)

        convert_deepmatcher_structure(os.path.join(Config.DATA_DIR, "deep_matcher", "valid.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableA.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableB.csv"),
                                      os.path.join(Config.DATA_DIR, "dev.tsv"),
                                      abt_buy_join_left, abt_buy_join_right)

    if Config.DATA_DIR.endswith("company"):
        convert_deepmatcher_structure(os.path.join(Config.DATA_DIR, "deep_matcher", "train.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableA.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableB.csv"),
                                      os.path.join(Config.DATA_DIR, "train.tsv"),
                                      company_join(), company_join)

        convert_deepmatcher_structure(os.path.join(Config.DATA_DIR, "deep_matcher", "test.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableA.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableB.csv"),
                                      os.path.join(Config.DATA_DIR, "test.tsv"),
                                      company_join, company_join)

        convert_deepmatcher_structure(os.path.join(Config.DATA_DIR, "deep_matcher", "valid.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableA.csv"),
                                      os.path.join(Config.DATA_DIR, "deep_matcher", "tableB.csv"),
                                      os.path.join(Config.DATA_DIR, "dev.tsv"),
                                      company_join, company_join)
