import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from src.utils.logger import LOGGER


def get_envs() -> Tuple[int, int, int]:
    """retrieves the PyTorch needed environments related to the distributed training setup

    :returns: a tuple containing the values of the environment variables LOCAL_RANK, RANK, and WORLD_SIZE
    """
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return local_rank, rank, world_size


def select_device(device: str) -> torch.device:
    """selects the device for training based on the provided device string

    :param device: (str) the device string specifying the device(s) to be used for training.
        possible values are 'cpu' for CPU, or a comma-separated list of GPU device IDs.
    :returns: (torch.device) the selected device for training
    :raises: (AssertionError) if the specified GPU device IDs are not available
    """
    if device == 'cpu':  # if CPU is specified
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force PyTorch to use CPU
        LOGGER.info('Using CPU for training...')
    elif device:  # if GPUs are specified, it should be a comma-separated list of GPU device IDs
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # specifying the GPU device IDs
        assert torch.cuda.is_available(), 'This PyTorch version does not support GPU. Please use CPU for training.'
        num_devices = len(device.strip().split(','))  # count the number of devices
        LOGGER.info(f'Using {num_devices} GPUs for training...')

    cuda = device != 'cpu' and torch.cuda.is_available()  # check if CUDA is available and the device is not CPU
    # when CUDA is enabled, just return the first CUDA device as `torch.device` object
    return torch.device('cuda:0' if cuda else 'cpu')


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """set the random state for reproducibility

    :param seed: (int) the seed value to be set
    :param deterministic: (bool) whether to enable deterministic mode or not
    """
    random.seed(seed)  # python's built-in random seed
    np.random.seed(seed)  # NumPy's random seed
    torch.manual_seed(seed)  # PyTorch's random seed
    if deterministic:
        # if deterministic mode is enabled, we want to ensure that all operatio
        # that use random numbers are deterministic. this can slow down the training
        # process, but it ensures that the results are reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # if deterministic mode is disabled, allowing non-deterministic operations
        cudnn.deterministic = False
        cudnn.benchmark = True
