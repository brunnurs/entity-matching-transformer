import logging
import os

from pytorch_transformers import BertTokenizer

from config import read_arguments_train, read_arguments_prediction
from data_representation import DeepMatcherProcessor, QqpProcessor
from logging_customized import setup_logging
from src.data_loader import load_data, DataType
from src.model import load_model
from src.prediction import predict
from torch_initializer import initialize_gpu_seed

setup_logging()

if __name__ == "__main__":
    args = read_arguments_prediction()

    device, n_gpu = initialize_gpu_seed(args.seed)

    model, tokenizer = load_model(os.path.join(args.model_output_dir, args.trained_model_for_prediction), args.do_lower_case)
    model.to(device)

    if tokenizer:
        logging.info("Loaded pretrained model and tokenizer from {}".format(args.trained_model_for_prediction))
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        logging.info("Loaded pretrained model from {} but no fine-tuned tokenizer found, therefore use the standard tokenizer."
                     .format(args.trained_model_for_prediction))

    if args.data_processor == "QqpProcessor":
        processor = QqpProcessor()
    else:
        # this is the default as it works for all data sets of the deepmatcher project.
        processor = DeepMatcherProcessor()

    test_examples = processor.get_test_examples(args.data_path)

    logging.info("loaded {} test examples".format(len(test_examples)))
    test_data_loader = load_data(test_examples,
                                 processor.get_labels(),
                                 tokenizer,
                                 args.max_seq_length,
                                 args.test_batch_size,
                                 DataType.TEST)

    simple_accuracy, f1, classification_report, predictions = predict(model, device, test_data_loader)
    logging.info("Prediction done for {} examples.F1: {}, Simple Accuracy: {}".format(len(test_data_loader), f1, simple_accuracy))

    logging.info(classification_report)

    logging.info(predictions)
