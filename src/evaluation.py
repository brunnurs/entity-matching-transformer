import logging
import os
import numpy as np

import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from logging_customized import setup_logging

setup_logging()


class Evaluation:

    def __init__(self, evaluation_data_loader, experiment_name, model_output_dir, n_labels):
        self.evaluation_data_loader = evaluation_data_loader
        self.n_labels = n_labels
        self.output_path = os.path.join(model_output_dir, experiment_name, "eval_results.txt")

    def evaluate(self, model, device, epoch, ):
        model.eval()
        nb_eval_steps, eval_loss = 0, 0

        all_logits = np.empty([0, self.n_labels])
        all_label_ids = np.empty(0)

        for input_ids, input_mask, segment_ids, label_ids in tqdm(self.evaluation_data_loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, self.n_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            logits_numpy = logits.detach().cpu().numpy()
            labels_numpy = label_ids.to('cpu').numpy()

            all_logits = np.concatenate([all_logits, logits_numpy])
            all_label_ids = np.concatenate([all_label_ids, labels_numpy])

        eval_loss = eval_loss / nb_eval_steps

        # remember, the logits are simply the output from the last layer, without applying an activation function (e.g. sigmoid).
        # for a simple classification this is also not necessary, we just take the index of the neuron with the maximal output.
        predicted_class = np.argmax(all_logits, axis=1)
        report = classification_report(all_label_ids, predicted_class)

        result = {'eval_loss': eval_loss}

        with open(self.output_path, "a+") as writer:
            tqdm.write("***** Eval results after epoch {} *****".format(epoch))
            writer.write("***** Eval results after epoch {} *****\n".format(epoch))
            for key in sorted(result.keys()):
                tqdm.write("{}: {}".format(key, str(result[key])))
                writer.write("{}: {}\n".format(key, str(result[key])))

            tqdm.write(report)
            writer.write(report + "\n")
