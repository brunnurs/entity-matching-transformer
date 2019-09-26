import logging
import sys


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        datefmt="%H:%M:%S",
                        stream=sys.stdout)

    logging.getLogger('bert-classifier-entity-matching')
