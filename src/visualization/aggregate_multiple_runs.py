import argparse
from typing import List

import pandas as pd

from pathlib import Path

from pandas import DataFrame
from tensorboardX import SummaryWriter


def load_data_frames(data_dir: str, naming_pattern: str) -> List[DataFrame]:
    input_path = Path.cwd() / data_dir

    data: List[DataFrame] = []
    for f in input_path.iterdir():
        if f.match("*{}*".format(naming_pattern)) and f.is_file():
            data.append(pd.read_csv(f.absolute()))

    return data


def average_data(data: DataFrame):
    data_without_steps = data.drop('Step', axis=1)
    data['Mean'] = data_without_steps.mean(axis=1)

    return data


def validate_data(data: List[DataFrame]):
    first_series_steps = data[0]['Step']

    f1_series = {'Step': first_series_steps}  # we wanna add the steps to be able to display the series in tensorboard.

    for idx, df in enumerate(data):
        if df['Step'].equals(first_series_steps):
            f1_series[idx] = df['Value']
        else:
            raise ValueError('Step series do not fit together! Series-Index: {}'.format(idx))

    return pd.DataFrame(f1_series)


def write_to_output_in_paper_format(averaged: DataFrame):
    for idx, average_value in enumerate(averaged['Mean']):
        print('\t({},{})'.format(idx, average_value * 100))

    print('Max value is: {}'.format(averaged['Mean'].max()))
    # how to visualize the mean values in an easy way:
    # (1) open averaged series in SciView.
    # (2) remove the first values, which will make the color-scheme useless: averaged[1:].Mean


def write_in_tensorboard_format(averaged: DataFrame, output_dir: str):
    output_path = Path.cwd() / output_dir
    writer = SummaryWriter(str(output_path))

    for average_value, step in zip(averaged['Mean'], averaged['Step']):
        writer.add_scalar('eval_f1_score', average_value, step)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate multiple validation runs with following arguments')

    parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument('--output_dir', default=None, type=str, required=True)
    parser.add_argument('--naming_pattern', default=None, type=str, required=True)

    args = parser.parse_args()

    data_raw = load_data_frames(args.data_dir, args.naming_pattern)
    data_validated = validate_data(data_raw)
    data_averaged = average_data(data_validated)
    write_to_output_in_paper_format(data_averaged)
    write_in_tensorboard_format(data_averaged, args.output_dir)
