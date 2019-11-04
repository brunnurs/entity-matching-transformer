import logging
import os
from datetime import datetime

from config import read_arguments_train, write_config_to_file, Config
from logging_customized import setup_logging
from src.data_loader import load_data, DataType
from src.data_representation import DeepMatcherProcessor, QqpProcessor
from src.evaluation import Evaluation
from src.model import save_model
from src.optimizer import build_optimizer
from torch_initializer import initialize_gpu_seed
from training import train

setup_logging()


def create_experiment_folder(model_output_dir: str, model_type: str, data_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = "{}__{}__{}".format(data_dir.upper(), model_type.upper(), timestamp)

    output_path = os.path.join(model_output_dir, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    return experiment_name


if __name__ == "__main__":
    args = read_arguments_train()

    exp_name = create_experiment_folder(args.model_output_dir, args.model_type, args.data_dir)

    write_config_to_file(args, args.model_output_dir, exp_name)

    device, n_gpu = initialize_gpu_seed(args.seed)

    if args.data_processor == "QqpProcessor":
        processor = QqpProcessor()
    else:
        # this is the default as it works for all data sets of the deepmatcher project.
        processor = DeepMatcherProcessor()

    label_list = processor.get_labels()

    logging.info("training with {} labels: {}".format(len(label_list), label_list))

    config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    model.to(device)
    logging.info("initialized {}-model".format(args.model_type))

    train_examples = processor.get_train_examples(args.data_path)
    training_data_loader = load_data(train_examples,
                                     label_list,
                                     tokenizer,
                                     args.max_seq_length,
                                     args.train_batch_size,
                                     DataType.TRAINING, args.model_type)
    logging.info("loaded {} training examples".format(len(train_examples)))

    num_train_steps = len(training_data_loader) * args.num_epochs

    optimizer, scheduler = build_optimizer(model,
                                           num_train_steps,
                                           args.learning_rate,
                                           args.adam_eps,
                                           args.warmup_steps,
                                           args.weight_decay)
    logging.info("Built optimizer: {}".format(optimizer))

    eval_examples = processor.get_test_examples(args.data_path)
    evaluation_data_loader = load_data(eval_examples,
                                       label_list,
                                       tokenizer,
                                       args.max_seq_length,
                                       args.eval_batch_size,
                                       DataType.EVALUATION, args.model_type)

    evaluation = Evaluation(evaluation_data_loader, exp_name, args.model_output_dir, len(label_list), args.model_type)
    logging.info("loaded and initialized evaluation examples {}".format(len(eval_examples)))

    train(device,
          training_data_loader,
          model,
          optimizer,
          scheduler,
          evaluation,
          args.num_epochs,
          args.max_grad_norm,
          args.save_model_after_epoch,
          experiment_name=exp_name,
          output_dir=args.model_output_dir,
          model_type=args.model_type)

    save_model(model, exp_name, args.model_output_dir, tokenizer=tokenizer)
