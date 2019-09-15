from enum import Enum

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

from logging_customized import setup_logging
from src.feature_extraction import convert_examples_to_features
import logging

setup_logging()


class DataType(Enum):
    TRAINING = "Training"
    EVALUATION = "Evaluation"
    TEST = "Test"


def load_data(examples, label_list, tokenizer, max_seq_length, batch_size, data_type: DataType):
    logging.info("***** Convert Data to Features (Word-Piece Tokenizing) [{}] *****".format(data_type))
    features = convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            output_mode="classification",
                                            cls_token_segment_id=0)

    logging.info("***** Build PyTorch DataLoader with extracted features [{}] *****".format(data_type))
    logging.info("  Num examples = %d", len(examples))
    logging.info("  Batch size = %d", batch_size)
    logging.info("  Max Sequence Length = %d", max_seq_length)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if data_type == DataType.TRAINING:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    return DataLoader(data, sampler=sampler, batch_size=batch_size)
