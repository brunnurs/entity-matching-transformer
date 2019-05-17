import os
import torch

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, WEIGHTS_NAME, CONFIG_NAME


def get_pretrained_model(model_path, num_labels, cache_dir, model_state_dict=None):
    if model_state_dict:
        model = BertForSequenceClassification.from_pretrained(model_path,
                                                              num_labels=num_labels,
                                                              state_dict=model_state_dict)
    else:
        model = BertForSequenceClassification.from_pretrained(model_path,
                                                              cache_dir=cache_dir,
                                                              num_labels=num_labels)
    return model


def save_model(model, experiment_name, model_output_dir):
    output_path = os.path.join(model_output_dir, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    output_model_file = os.path.join(output_path, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(output_path, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())


def load_saved_model(experiment_name, model_output_dir, num_labels, ):
    saved_model_path = os.path.join(model_output_dir, experiment_name)
    model_state_dict = torch.load(os.path.join(saved_model_path, WEIGHTS_NAME))
    model = BertForSequenceClassification.from_pretrained(saved_model_path,
                                                          num_labels=num_labels,
                                                          state_dict=model_state_dict)

    return model
