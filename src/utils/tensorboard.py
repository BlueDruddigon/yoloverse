from typing import Any, List

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from src.utils.logger import LOGGER


def write_tblog(writer: SummaryWriter, epoch: int, results: List[float], lrs: List[Any], losses: torch.Tensor) -> None:
    """writes the training and validation metrics to TensorBoard

	:param writer: (SummaryWriter) the TensorBoard SummaryWriter object
	:param epoch: (int) the current epoch number
    :param results: (list) the validation results, including mAP@0.5 and mAP@0.5:0.95
	:param lrs: (list) the learning rates for different layers
	:param losses: (tensor) the training losses for different components
	"""
    writer.add_scalar('val/mAP@0.5', results[0], epoch + 1)
    writer.add_scalar('val/mAP@0.5:0.95', results[1], epoch + 1)

    writer.add_scalar('train/iou_loss', losses[0], epoch + 1)
    writer.add_scalar('train/dfl_loss', losses[1], epoch + 1)
    writer.add_scalar('train/cls_loss', losses[2], epoch + 1)

    writer.add_scalar('x/lr0', lrs[0], epoch + 1)
    writer.add_scalar('x/lr1', lrs[1], epoch + 1)
    writer.add_scalar('x/lr2', lrs[2], epoch + 1)


def write_tbimg(writer: SummaryWriter, imgs: np.ndarray | List[torch.Tensor], step: int, type: str = 'train') -> None:
    """writes images to TensorBoard

	:param writer: (SummaryWriter) the TensorBoard SummaryWriter object
	:param imgs: (list | np.ndarray) the images to be written
	:param step: (int) the current step or iteration number
	:param type: (str) the type of images. default: 'train'
	"""
    if type == 'train':  # check type of images to write
        # if the type is 'train', write the batch of images to the TensorBoard
        writer.add_image('train_batch', imgs, step + 1, dataformats='HWC')
    elif type == 'val':
        # if validation, add each image in the batch to TensorBoard
        for idx, img in enumerate(imgs):
            writer.add_image(f'val_img_{idx+1}', img, step + 1, dataformats='HWC')
    else:  # log a warning if the image type is unknown
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')
