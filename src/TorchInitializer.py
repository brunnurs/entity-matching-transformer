import random
import numpy as np
import torch
import logging


class TorchInitializer:
    def initialize_gpu_seed(self, seed: int):
        device, n_gpu = self._setup_gpu()

        self._init_seed_everywhere(seed, n_gpu)

        return device, n_gpu

    @staticmethod
    def _init_seed_everywhere(seed, n_gpu):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _setup_gpu():
        # Setup GPU parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        logging.info("We use the device: '{}' and {} gpu's. Important: distributed and 16-bits training "
                     "is currently not implemented! ".format(device, n_gpu))

        return device, n_gpu
