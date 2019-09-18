import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm


def predict(model, device, test_data_loader):
    nb_prediction_steps = 0
    predictions = None
    labels = None

    for batch in tqdm(test_data_loader, desc="Test"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            outputs = model(**inputs)
            _, logits = outputs[:2]

        nb_prediction_steps += 1

        if predictions is None:
            predictions = logits.detach().cpu().numpy()
            labels = inputs['labels'].detach().cpu().numpy()
        else:
            predictions = np.append(predictions, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs['labels'].detach().cpu().numpy(), axis=0)

    # remember, the logits are simply the output from the last layer, without applying an activation function (e.g. sigmoid).
    # for a simple classification this is also not necessary, we just take the index of the neuron with the maximal output.
    predicted_class = np.argmax(predictions, axis=1)

    simple_accuracy = (predicted_class == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=predicted_class)
    report = classification_report(labels, predicted_class)

    return simple_accuracy, f1, report, pd.DataFrame({'predictions': predicted_class, 'labels': labels})
