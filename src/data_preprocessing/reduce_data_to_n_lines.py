from src import write_first_n_lines_of_file

if __name__ == "__main__":
    write_first_n_lines_of_file("/home/ursin/development/binary-classification-bert/data/SST2/train.original.tsv",
                                "/home/ursin/development/binary-classification-bert/data/SST2/train.tsv",
                                10)

    write_first_n_lines_of_file("/home/ursin/development/binary-classification-bert/data/SST2/dev.original.tsv",
                                "/home/ursin/development/binary-classification-bert/data/SST2/dev.tsv",
                                10)
