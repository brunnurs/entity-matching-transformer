import logging

import torch
from tqdm import tqdm, trange

from logging_customized import setup_logging
from model import save_model

setup_logging()


def train(device, train_dataloader, model, optimizer, scheduler, evaluation, num_epocs, max_grad_norm, experiment_name, output_dir):
    logging.info("***** Running training *****")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    for epoch in trange(int(num_epocs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            global_step += 1

            # tqdm.write('lr: {}'.format(scheduler.get_lr()[0]))
            # tqdm.write('loss: {}'.format(tr_loss - logging_loss))
            logging_loss = tr_loss

        evaluation.evaluate(model, device, epoch)
        save_model(model, experiment_name, output_dir, epoch=epoch)
