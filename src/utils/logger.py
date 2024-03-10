import logging
import os
import shutil
from logging import Logger
from typing import Optional


def set_logging(name: Optional[str] = None) -> Logger:
    rank = int(os.getenv('RANK', default=-1))
    logging.basicConfig(format='%(message)s', level=logging.INFO if rank in (-1, 0) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)
