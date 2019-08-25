import argparse
import os
import logging

import jsons

from logging_customized import setup_logging

setup_logging()


# this implementation is stolen from https://stackoverflow.com/a/11517201/1081551, which is based on the python documentation.
class Singleton(object):
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        pass


class Config(Singleton):
    DATA_DIR = "data/company"
    PRE_TRAINED_MODEL_DIR = "pre_trained_model"
    PRE_TRAINED_MODEL_BERT_BASE_UNCASED = os.path.join(PRE_TRAINED_MODEL_DIR, "bert-base-uncased")
    MODEL_OUTPUT_DIR = "experiments"
    MODEL_NAME = "ABT_BUY"
    TRAINED_MODEL_FOR_PREDICTION = "SST2_1553095240"
    MAX_SEQ_LENGTH = 128
    DO_LOWER_CASE = True
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    ADAM_EPS = 1e-6
    WARMUP_STEPS = 0
    MAX_GRAD_NORM = 1.0
    WEIGHT_DECAY = 0.0

    NUM_EPOCHS = 30.0
    SEED = 42
    LOSS_SCALE = 128

    DATA_PROCESSOR = "DeepMatcherProcessor"

    @classmethod
    def dump_config_to_json_file(cls, experiment_name):
        config_path = os.path.join(cls.MODEL_OUTPUT_DIR, experiment_name, "config.json")

        with open(config_path, 'w') as outfile:
            outfile.write(jsons.dumps(Config()))

    @classmethod
    def set_arguments_to_config(cls):
        parser = argparse.ArgumentParser(description='Example with long option names')

        parser.add_argument('--data_dir', action="store", dest="DATA_DIR", type=str)
        parser.add_argument('--model_name', action="store", dest="MODEL_NAME", type=str)
        parser.add_argument('--max_seq_length', action="store", dest="MAX_SEQ_LENGTH", type=int)
        parser.add_argument('--train_batch_size', action="store", dest="TRAIN_BATCH_SIZE", type=int)
        parser.add_argument('--eval_batch_size', action="store", dest="EVAL_BATCH_SIZE", type=int)
        parser.add_argument('--num_epochs', action="store", dest="NUM_EPOCHS", type=float)
        parser.add_argument('--data_processor', action="store", dest="DATA_PROCESSOR", type=str)

        results = parser.parse_args()

        logging.info("*** parse configuration from command line ***")
        for argument in vars(results):
            cfg = Config()

            if getattr(results, argument):
                logging.info("Overwrite configuration {} with value {} from command line".format(argument,
                                                                                          getattr(results, argument)))
                setattr(cfg, argument, getattr(results, argument))
