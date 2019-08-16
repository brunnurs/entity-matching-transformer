import logging

import torch
from tqdm import tqdm, trange

from logging_customized import setup_logging

setup_logging()


class Training:

    def fit(self, device, train_dataloader, model, optimizer, scheduler, evaluation, num_epocs, max_grad_norm):

        logging.info("***** Running training *****")

        global_step = 0
        for i_ in trange(int(num_epocs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                outputs = model(input_ids, segment_ids, input_mask, label_ids)
                loss = outputs[0]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

            tqdm.write('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
            evaluation.evaluate(model, device, i_)
