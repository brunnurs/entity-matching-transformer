import argparse
import json
import os
import logging

from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer, XLNetTokenizer, \
    XLNetForSequenceClassification, XLNetConfig, XLMForSequenceClassification, XLMConfig, XLMTokenizer, \
    RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig

from logging_customized import setup_logging

setup_logging()


class Config():
    DATA_PREFIX = "data"
    EXPERIMENT_PREFIX = "experiments"

    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    }

    # PRE_TRAINED_MODEL_DIR = "pre_trained_model"
    # PRE_TRAINED_MODEL_BERT_BASE_UNCASED = os.path.join(PRE_TRAINED_MODEL_DIR, "bert-base-uncased")
    # MODEL_OUTPUT_DIR = "experiments"
    # MODEL_NAME = "__please_config_me__"
    # TRAINED_MODEL_FOR_PREDICTION = "__please_config_me__"
    # MAX_SEQ_LENGTH = 128
    # DO_LOWER_CASE = True
    # TRAIN_BATCH_SIZE = 16
    # EVAL_BATCH_SIZE = 16
    # TEST_BATCH_SIZE = 16
    # LEARNING_RATE = 2e-5
    # ADAM_EPS = 1e-8
    # WARMUP_STEPS = 0
    # MAX_GRAD_NORM = 1.0
    # WEIGHT_DECAY = 0.0
    #
    # NUM_EPOCHS = 30.0
    # SEED = 42
    # LOSS_SCALE = 128
    #
    # DATA_PROCESSOR = "DeepMatcherProcessor"
    #
    # SAVE_MODEL_AFTER_EPOCH = False


def write_config_to_file(args, model_output_dir: str, experiment_name: str):
    config_path = os.path.join(model_output_dir, experiment_name, "args.json")

    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def read_arguments_train():
    parser = argparse.ArgumentParser(description='Run training with following arguments')

    parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument('--data_processor', default=None, type=str, required=True)
    parser.add_argument('--model_name_or_path', default="pre_trained_model/bert-base-uncased", type=str, required=True)
    parser.add_argument('--model_type', default='bert', type=str)
    parser.add_argument('--do_lower_case', action='store_true', default=True)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--num_epochs', default=3.0, type=float)
    parser.add_argument('--save_model_after_epoch', action='store_true')
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    args.data_path = os.path.join(Config.DATA_PREFIX, args.data_dir)
    args.model_output_dir = Config.EXPERIMENT_PREFIX

    logging.info("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        logging.info("argument: {}={}".format(argument, getattr(args, argument)))

    return args


def read_arguments_prediction():
    parser = argparse.ArgumentParser(description='Run testing with the following arguments')

    parser.add_argument('--trained_model_for_prediction', action="store", required=True, type=str)
    parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument('--data_processor', default=None, type=str, required=True)
    parser.add_argument('--do_lower_case', action='store_true', default=True)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--test_batch_size', default=8, type=int)

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    args.data_path = os.path.join(Config.DATA_PREFIX, args.data_dir)
    args.model_output_dir = Config.EXPERIMENT_PREFIX

    logging.info("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        logging.info("argument: {}={}".format(argument, getattr(args, argument)))

    return args
