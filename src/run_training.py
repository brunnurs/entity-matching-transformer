import logging
import os
import time

from pytorch_pretrained_bert import BertTokenizer

from logging_customized import setup_logging
from src.TorchInitializer import TorchInitializer
from src.config import Config
from src.data_loader import load_data, DataType
from src.data_representation import DeepMatcherProcessor
from src.evaluation import Evaluation
from src.model import get_pretrained_model, save_model
from src.optimizer import build_optimizer
from training import Training

setup_logging()


def create_experiment_folder():
    timestamp = str(int(time.time()))
    experiment_name = "{}_{}".format(Config.MODEL_NAME, timestamp)

    output_path = os.path.join(Config.MODEL_OUTPUT_DIR, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    return experiment_name


if __name__ == "__main__":
    exp_name = create_experiment_folder()

    device, n_gpu = TorchInitializer().initialize_gpu_seed(Config.SEED)

    processor = DeepMatcherProcessor()
    label_list = processor.get_labels()

    logging.info("training with {} labels: {}".format(len(label_list), label_list))

    tokenizer = BertTokenizer.from_pretrained(Config.PRE_TRAINED_MODEL_BERT_BASE_UNCASED, do_lower_case=Config.DO_LOWER_CASE)

    train_examples = processor.get_train_examples(Config.DATA_DIR)
    logging.info("loaded {} training examples".format(len(train_examples)))

    model = get_pretrained_model(Config.PRE_TRAINED_MODEL_BERT_BASE_UNCASED, len(label_list), Config.PRE_TRAINED_MODEL_CACHE_DIR)
    model.to(device)
    logging.info("initialized BERT-model")

    num_train_steps = int(len(train_examples) / Config.TRAIN_BATCH_SIZE) * Config.NUM_EPOCHS

    optimizer = build_optimizer(model, num_train_steps, Config.LEARNING_RATE, Config.WARMUP_PROPORTION, Config.WEIGHT_DECAY)
    logging.info("Built optimizer: {}".format(optimizer))

    eval_examples = processor.get_dev_examples(Config.DATA_DIR)
    evaluation_data_loader = load_data(eval_examples, label_list, tokenizer, Config.MAX_SEQ_LENGTH, Config.EVAL_BATCH_SIZE, DataType.EVALUATION)

    evaluation = Evaluation(evaluation_data_loader, exp_name, Config.MODEL_OUTPUT_DIR, len(label_list))
    logging.info("loaded and initialized evaluation examples {}".format(len(eval_examples)))

    training = Training()
    training_data_loader = load_data(train_examples, label_list, tokenizer, Config.MAX_SEQ_LENGTH, Config.TRAIN_BATCH_SIZE, DataType.TRAINING)

    training.fit(device, training_data_loader, model, optimizer, evaluation, Config.NUM_EPOCHS)

    save_model(model, exp_name, Config.MODEL_OUTPUT_DIR)

