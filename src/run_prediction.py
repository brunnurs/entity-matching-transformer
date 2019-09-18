import logging
import os

from pytorch_transformers import BertTokenizer

from config import Config
from data_representation import DeepMatcherProcessor, QqpProcessor
from logging_customized import setup_logging
from src.data_loader import load_data, DataType
from src.model import load_model
from src.prediction import predict
from torch_initializer import initialize_gpu_seed

setup_logging()

if __name__ == "__main__":
    Config().set_arguments_to_config()

    device, n_gpu = initialize_gpu_seed(Config().SEED)

    model, tokenizer = load_model(os.path.join(Config().MODEL_OUTPUT_DIR, Config().TRAINED_MODEL_FOR_PREDICTION), Config().DO_LOWER_CASE)
    model.to(device)

    if tokenizer:
        logging.info("Loaded pretrained model and tokenizer from {}".format(Config().TRAINED_MODEL_FOR_PREDICTION))
    else:
        tokenizer = BertTokenizer.from_pretrained(Config().PRE_TRAINED_MODEL_BERT_BASE_UNCASED, do_lower_case=Config().DO_LOWER_CASE)
        logging.info("Loaded pretrained model from {} but no fine-tuned tokenizer found, therefore use the standard tokenizer."
                     .format(Config().TRAINED_MODEL_FOR_PREDICTION))

    if Config().DATA_PROCESSOR == "QqpProcessor":
        processor = QqpProcessor()
    else:
        # this is the default as it works for all data sets of the deepmatcher project.
        processor = DeepMatcherProcessor()

    test_examples = processor.get_test_examples(Config().DATA_DIR)

    logging.info("loaded {} test examples".format(len(test_examples)))
    test_data_loader = load_data(test_examples, processor.get_labels(), tokenizer, Config().MAX_SEQ_LENGTH, Config().TEST_BATCH_SIZE, DataType.TEST)

    simple_accuracy, f1, classification_report, predictions = predict(model, device, test_data_loader)
    logging.info("Prediction done for {} examples.F1: {}, Simple Accuracy: {}".format(len(test_data_loader), f1, simple_accuracy))

    logging.info(classification_report)

    logging.info(predictions)
