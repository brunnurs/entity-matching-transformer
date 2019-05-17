import logging
import random
import csv


def create_validation_file_from_train_file(path_train, path_validation, percentage_of_train_file):
    with open(path_train) as just_to_get_the_length:
        length_train_file = len(list(just_to_get_the_length))

    amount_of_lines_for_val = int(length_train_file * percentage_of_train_file)
    samples_for_validation = random.sample([*range(0, length_train_file)], amount_of_lines_for_val)

    with open(path_train) as src, open(path_validation, "w+") as validation_dest, open(path_train + ".new",
                                                                                       "w+") as training_dest:

        reader = csv.reader(src, delimiter=",")
        writer_validation = csv.writer(validation_dest, delimiter=',')
        writer_train_new = csv.writer(training_dest, delimiter=',')

        for idx, line in enumerate(reader):
            if idx != 0:
                if idx in samples_for_validation:
                    writer_validation.writerow(line)
                else:
                    writer_train_new.writerow(line)
            else:
                writer_validation.writerow(line)
                writer_train_new.writerow(line)

            if idx % 1000 == 0:
                logging.info("Writing new CSV's. Line:{}".format(idx))


def write_first_n_lines_of_file(path_src, path_dest, n_lines):

    with open(path_src) as src, open(path_dest, "w+") as dest:

        reader = csv.reader(src, delimiter=",")
        writer_dest = csv.writer(dest, delimiter=',')

        for idx, line in enumerate(reader):
            if idx < n_lines:
                writer_dest.writerow(line)
            else:
                break

        logging.info("created new CSV ({}) with first {} lines of {}.".format(path_dest, n_lines, path_src))
