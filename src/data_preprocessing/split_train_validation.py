from src import create_validation_file_from_train_file

if __name__ == "__main__":
    create_validation_file_from_train_file("/home/ursin/development/binary-classification-bert/data/toxic_comments/train.csv",
                                           "/home/ursin/development/binary-classification-bert/data/toxic_comments/val.csv", 0.1)