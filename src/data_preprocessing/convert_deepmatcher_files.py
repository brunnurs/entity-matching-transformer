import csv
import os

import pandas as pd
import logging

import constants as C
from config import Config
from logging_customized import setup_logging

setup_logging()


def convert_abt_buy_files(filename_matches: str, filename_table_a: str,
                          filename_table_b: str, filename_destination: str):

    df_table_a = pd.read_csv(filename_table_a, sep=",", encoding="UTF-8")
    df_table_b = pd.read_csv(filename_table_b, sep=",", encoding="UTF-8")

    with open(filename_matches) as src, open(filename_destination, "w+") as dest:
        reader = csv.DictReader(src, delimiter=",")
        writer = csv.writer(dest, delimiter="\t")

        writer.writerow([C.INDEX_KEY, C.TEXT_LEFT, C.TEXT_RIGHT, C.LABEL])

        for idx, line in enumerate(reader):
            left = df_table_a.iloc[int(line['ltable_id'])]
            right = df_table_b.iloc[int(line['rtable_id'])]

            # left_joined_attributes = " ".join([left['name'], left['description']])
            left_joined_attributes = left['description']

            if right.isnull()['description']:
                right_joined_attributes = right['name']
            else:
                right_joined_attributes = " ".join([right['name'], right['description']])

            writer.writerow([idx, left_joined_attributes, right_joined_attributes, line['label']])

    logging.info("Created new tsv for abt_buy. Filename: {}".format(filename_destination))


if __name__ == "__main__":
    convert_abt_buy_files(os.path.join(Config.DATA_DIR, "deep_matcher", "train.csv"),
                          os.path.join(Config.DATA_DIR, "deep_matcher", "tableA.csv"),
                          os.path.join(Config.DATA_DIR, "deep_matcher", "tableB.csv"),
                          os.path.join(Config.DATA_DIR, "train.tsv"))

    convert_abt_buy_files(os.path.join(Config.DATA_DIR, "deep_matcher", "test.csv"),
                          os.path.join(Config.DATA_DIR, "deep_matcher", "tableA.csv"),
                          os.path.join(Config.DATA_DIR, "deep_matcher", "tableB.csv"),
                          os.path.join(Config.DATA_DIR, "test.tsv"))

    convert_abt_buy_files(os.path.join(Config.DATA_DIR, "deep_matcher", "valid.csv"),
                          os.path.join(Config.DATA_DIR, "deep_matcher", "tableA.csv"),
                          os.path.join(Config.DATA_DIR, "deep_matcher", "tableB.csv"),
                          os.path.join(Config.DATA_DIR, "dev.tsv"))
