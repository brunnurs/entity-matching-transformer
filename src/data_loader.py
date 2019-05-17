from enum import Enum

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

from src.feature_extraction import convert_examples_to_features
import logging


class DataType(Enum):
    TRAINING = "Training"
    EVALUATION = "Evaluation"
    TEST = "Test"


def load_data(examples, label_list, tokenizer, max_seq_length, batch_size, data_type: DataType):
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)

    logging.info("***** Load Data for {}*****".format(data_type))
    logging.info("  Num examples = %d", len(examples))
    logging.info("  Batch size = %d", batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id)

    if data_type == DataType.TRAINING:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    return DataLoader(data, sampler=sampler, batch_size=batch_size)
