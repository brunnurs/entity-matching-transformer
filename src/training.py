import logging

import torch
from tqdm import tqdm, trange

from logging_customized import setup_logging
from model import save_model
from tensorboardX import SummaryWriter

setup_logging()


def train(device, train_dataloader, model, optimizer, scheduler, evaluation, num_epocs, max_grad_norm, experiment_name, output_dir):
    logging.info("***** Run training *****")
    tb_writer = SummaryWriter()

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

            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
            logging_loss = tr_loss

        eval_results = evaluation.evaluate(model, device, epoch)
        for key, value in eval_results.items():
            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        save_model(model, experiment_name, output_dir, epoch=epoch)

    tb_writer.close()