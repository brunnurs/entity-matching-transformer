import logging

from tqdm import tqdm, trange


class Training:

    def __call__(self, device, train_dataloader, model, optimizer, evaluation, num_epocs):
        self.fit(device, train_dataloader, model, optimizer, evaluation, num_epocs)

    def fit(self, device, train_dataloader, model, optimizer, evaluation, num_epocs):

        logging.info("***** Running training *****")

        global_step = 0
        for i_ in trange(int(num_epocs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            tqdm.write('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
            tqdm.write('Eval after epoc {}'.format(i_ + 1))
            evaluation.evaluate(model, device)
