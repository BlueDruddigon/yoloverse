import glob
import math
import os
from pathlib import Path
from typing import Sequence, Union

from .logger import LOGGER


def increment_name(path: Union[str, Path]) -> Path:
    """increments the name of a file or directory if it already exists

    :param path: (str | Path) the original path
    :return: (Path) the incremented path
    """
    path = Path(path)  # convert to Path object if it's not
    sep = ''  # separator
    if path.exists():  # check existence
        # separate the file's extension and path for both file and directory
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):  # number to be added to the name
            p = f'{path}{sep}{n}{suffix}'  # update new name
            if not os.path.exists(p):  # exit if the new file/directory is not exist
                break
        path = Path(p)  # set the path to the new path
    return path


def find_latest_checkpoint(search_dir: str = '.') -> str:
    """ find the most recent saved checkpoint from `search_dir` """
    # grab all last checkpoints in the search directory recursively and only take the last updated one
    checkpoint_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(checkpoint_list, key=os.path.getctime) if checkpoint_list else ''


def make_divisible(x: int, divisor: int) -> int:
    """adjusts the given number to be divisible by the specified divisor by rounding up the division result

    :param x: (int) the original number
    :param divisor: (int) the divisor
    :return: (int) the adjusted number
    """
    return math.ceil(x / divisor) * divisor


def check_image_size(
  image_size: Union[int, Sequence[int]],
  stride: int = 32,
  floor: int = 0,
  return_same_type: bool = True
) -> Union[int, Sequence[int]]:
    """check image size to be divisible by stride and not smaller than floor,
    and adjust the size if necessary

    :param image_size: (int | list) the original image size
    :param stride: (int) the stride value of the model. default: 32
    :param floor: (int) the minimum allowable size. default: 0
    :param return_same_type: (bool) whether to return the same type as the input. default: True
    :return: (int | list) the adjusted image size. return the original if no adjustment needed
    """
    if isinstance(image_size, int):  # handle integer value for the original image size
        # compute new size based on the image size, stride and floor values
        new_size = max(make_divisible(image_size, int(stride)), floor)
    elif isinstance(image_size, (list, tuple)):  # same as int, but make it be a list of int
        new_size = [max(make_divisible(x, int(stride)), floor) for x in image_size]
    else:
        raise Exception(f'Unsupported type of img_size: {type(image_size)}')
    if new_size != image_size:  # if the size is updated
        LOGGER.warning(f'--image-size {image_size} must be multiple of max stride {stride}, updating to {new_size}')

    if return_same_type:
        return new_size

    # force return as a list of 2 integers
    return new_size if isinstance(new_size, (list, tuple)) else [new_size] * 2
