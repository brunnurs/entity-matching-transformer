import csv
import logging

from logging_customized import setup_logging

setup_logging()


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


if __name__ == "__main__":
    write_first_n_lines_of_file("data/dirty_walmart_amazon/train.large.tsv", "data/dirty_walmart_amazon/train.tsv", 10)
    write_first_n_lines_of_file("data/dirty_walmart_amazon/test.large.tsv", "data/dirty_walmart_amazon/test.tsv", 10)
    write_first_n_lines_of_file("data/dirty_walmart_amazon/dev.large.tsv", "data/dirty_walmart_amazon/dev.tsv", 10)
