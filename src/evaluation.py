import logging
import os

import torch
from tqdm import tqdm

from src.metrics import accuracy


class Evaluation:

    def __init__(self, evaluation_data_loader, experiment_name, model_output_dir):
        self.evaluation_data_loader = evaluation_data_loader

        self.output_path = os.path.join(model_output_dir, experiment_name, "eval_results.txt")

    def evaluate(self, model, device):
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for input_ids, input_mask, segment_ids, label_ids in tqdm(self.evaluation_data_loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy}

        with open(self.output_path, "a+") as writer:
            tqdm.write("***** Eval results *****")
            writer.write("***** Eval results *****\n")
            for key in sorted(result.keys()):
                tqdm.write("{}: {}".format(key, str(result[key])))
                writer.write("{}: {}\n".format(key, str(result[key])))
