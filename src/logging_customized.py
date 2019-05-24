import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        datefmt="%H:%M:%S")

    logging.getLogger('bert-classifier-entity-matching')
