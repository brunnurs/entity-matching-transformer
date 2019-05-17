import logging

from pytorch_pretrained_bert import BertTokenizer
from sklearn.metrics import classification_report

from src.TorchInitializer import TorchInitializer
from src.config import Config
from src.data_loader import load_data, DataType
from src.data_representation import Sst2Processor
from src.model import load_saved_model
from src.prediction import predict

if __name__ == "__main__":
    device, n_gpu = TorchInitializer().initialize_gpu_seed(Config.SEED)

    processor = Sst2Processor()
    label_list = processor.get_labels()

    logging.info("Predict with {} labels: {}".format(len(label_list), label_list))

    tokenizer = BertTokenizer.from_pretrained(Config.PRE_TRAINED_MODEL_BERT_BASE_UNCASED,
                                              do_lower_case=Config.DO_LOWER_CASE)

    test_examples = processor.get_dev_examples(Config.DATA_DIR)
    logging.info("loaded {} test examples (take care: we use the evaluation-data here as no labels for test are "
                 "available!)".format(len(test_examples)))
    test_data_loader = load_data(test_examples, label_list, tokenizer, Config.MAX_SEQ_LENGTH,
                                       Config.EVAL_BATCH_SIZE, DataType.TEST)

    model = load_saved_model(Config.TRAINED_MODEL_FOR_PREDICTION, Config.MODEL_OUTPUT_DIR, len(label_list))
    model.to(device)

    prediction = predict(test_examples, test_data_loader, model, device)
    print("Prediction done! Result-Shape: {}".format(prediction.shape))

    print(prediction[:10])

    print(classification_report(list(prediction['true_label']), list(prediction['predicted_label'])))



