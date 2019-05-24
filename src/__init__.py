import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(message)s',
                    datefmt="%H:%M:%S")

logging.getLogger('bert-classifier')
