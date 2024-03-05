from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch.distributed as dist
from torch.utils.data import dataloader, distributed

from src.utils.logger import LOGGER
from src.utils.torch_utils import torch_distributed_zero_first

from .datasets import CPU_COUNT, TrainValDataset

assert CPU_COUNT is not None  # for safety


class TrainValDataLoader(dataloader.DataLoader):
    def __init__(self, *args, **kwargs) -> None:
        """initializes the DataLoader object

        :param *args: Variable length argument list.
        :param **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)  # init default attributes
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))  # update the batch sampler
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """ get the length of the dataset """
        return len(list(self.batch_sampler.sampler))

    def __iter__(self) -> Iterator[List[int]]:
        """ returns an iterator that yields batches of data """
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    def __init__(self, sampler: Optional[Iterable[List]]) -> None:
        """a sampler that repeats the given sampler indefinitely

        :param sampler: (iterable) the sampler to repeat
        """
        self.sampler = sampler

    def __iter__(self) -> Iterator[List[int]]:
        """ returns an iterator that yields samples from the data loader """
        if self.sampler is None:
            return

        while True:
            yield from iter(self.sampler)


def create_dataloader(
  path: str,
  img_size: int,
  batch_size: int,
  stride: Union[float, int],
  hyp: Optional[Dict[str, float]] = None,
  augment: bool = False,
  check_images: bool = False,
  check_labels: bool = False,
  pad: float = 0.0,
  rect: bool = False,
  rank: int = -1,
  workers: int = 8,
  shuffle: bool = False,
  data_dict: Optional[dict] = None,
  task: str = 'Train',
  specific_shape: bool = False,
  height: int = 1088,
  width: int = 1920,
  cache_ram: bool = False,
  save_valid_record: bool = False
) -> Tuple[TrainValDataLoader, TrainValDataset]:
    """create a data loader for training or validation

    :param path: (str) the path to the dataset
    :param img_size: (int): the size of the input images
    :param batch_size: (int): the batch size
    :param stride: (int | float): the stride for the dataset
    :param hyp: (dict): hyperparameters for data augmentation. default: None
    :param augment: (bool): whether to apply data augmentation. default: False
    :param check_images: (bool): whether to check the images. default: False
    :param check_labels: (bool): whether to check the labels. default: False
    :param pad: (float): padding value. default: 0.0
    :param rect: (bool): whether to use rectangular images. default: False
    :param rank: (int): the rank of the process in distributed training. default: -1
    :param workers: (int): the number of worker processes. default: 8
    :param shuffle: (bool): whether to shuffle the dataset. default: False
    :param data_dict: (Optional[dict]): additional data dictionary. default: None
    :param task: (str): the task type ('train' or 'val'). default: 'train'
    :param specific_shape: (bool): whether to use a specific shape for the images. default: False
    :param height: (int): the height of the images. default: 1088
    :param width: (int): the width of the images. default: 1920
    :param cache_ram: (bool): whether to cache data in RAM. default: False
    :param save_valid_record: (bool) whether to save the validation record. default: False
    :return: a tuple containing the data loader and the dataset
    """
    if rect and shuffle:  # handle inappropriate cases
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False

    # create the dataset within a distributed context
    with torch_distributed_zero_first(rank):
        dataset = TrainValDataset(
          path,
          img_size,
          batch_size,
          augment=augment,
          hyp=hyp,
          rect=rect,
          check_images=check_images,
          check_labels=check_labels,
          stride=int(stride),
          pad=pad,
          rank=rank,
          data_dict=data_dict,
          task=task,
          specific_shape=specific_shape,
          height=height,
          width=width,
          cache_ram=cache_ram,
          save_valid_record=save_valid_record
        )

    # adjust batch size based on the dataset's length
    batch_size = min(batch_size, len(dataset))
    workers = min([CPU_COUNT, batch_size if batch_size > 1 else 0, workers])  # number of workers
    drop_last = rect and dist.is_initialized() and dist.get_world_size() > 1  # determine whether to drop the last batch
    # create a distributed sampler if necessary
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)

    return TrainValDataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle and sampler is None,
      num_workers=workers,
      sampler=sampler,
      pin_memory=True,
      collate_fn=TrainValDataset.collate_fn
    ), dataset
