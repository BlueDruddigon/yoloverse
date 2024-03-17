import os
import os.path as osp
import shutil
from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import LOGGER
from src.utils.torch_utils import fuse_model


def load_state_dict(weights: str, model: nn.Module, map_location: Optional[str | torch.device] = None):
    """loads the state dictionary from a checkpoint file and assigns it to the model

    :param weights: (str) path to the checkpoint file
    :param model: (nn.Module) the model to load the state dictionary into
    :param map_location: (str | torch.device) specifies where to load the checkpoint on the device
    :returns: (nn.Module) the model with loaded state dict
    """
    ckpt = torch.load(weights, map_location=map_location)  # load checkpoint
    state_dict = ckpt['model'].float().state_dict()  # extract model's state dict as FP32
    model_state_dict = model.state_dict()  # current model's state dict
    # merges these two state dicts
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(state_dict, strict=False)  # load filtered state dict into the model
    del ckpt, state_dict, model_state_dict  # delete the temporary variables to free up memory
    return model


def load_checkpoint(weights: str, map_location: Optional[str | torch.device] = None, fuse: bool = True):
    """loads a checkpoint from the specified path

    :param weights: (str) path to the checkpoint file
    :param map_location: (str | torch.device) specifies where to load the checkpoint on the device
    :param fuse: (bool) whether to fuse the model
    :returns: (nn.Module) the loaded model
    """
    LOGGER.info(f'Loading checkpoint from {weights}')  # log the checkpoint path
    ckpt = torch.load(weights, map_location=map_location)  # load checkpoint
    # get the model from checkpoint and convert it to FP32. use EMA if available
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    if fuse:  # if fusing, fuse the model and set it to evaluation mode
        LOGGER.info('\nFusing model...')
        model = fuse_model(model).eval()  # fuse and evaluation mode
    else:  # if not fusing, set the model to evaluation mode
        model = model.eval()

    return model  # return the loaded model


def save_checkpoint(ckpt: dict, is_best: bool, save_dir: str, model_name: str = '') -> None:
    """save a checkpoint dictionary to a file

    :param ckpt: (dict) the checkpoint dictionary to be saved
    :param is_best: (bool) flag indicating if this checkpoint is the best one
    :param save_dir: (str) the directory where the checkpoint file will be saved
    :param model_name: (str) the name of the model
    """
    if not osp.exists(save_dir):  # create directory if not exist
        os.makedirs(save_dir, exist_ok=True)
    fname = osp.join(save_dir, f'{model_name}.pt')  # path to save the checkpoint
    torch.save(ckpt, fname)
    if is_best:  # if `is_best`, copy the checkpoint to `best_ckpt.pt`
        best_fname = osp.join(save_dir, 'best_ckpt.pt')
        shutil.copyfile(fname, best_fname)


def strip_optimizer(ckpt_dir: str, epoch: int) -> None:
    """strips the optimizer and unnecessary keys from a checkpoint file and saves the modified checkpoint

    :param ckpt_dir: (str) the directory where the checkpoint file is located
    :param epoch: (int) the epoch number to be set in the modified checkpoint
    """
    for s in ('best', 'last'):  # only strip `best` and `last`
        ckpt_path = osp.join(ckpt_dir, f'{s}_ckpt.pt')
        if not osp.exists(ckpt_path):  # pass if the checkpoint file does not exist
            continue
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))  # load checkpoint
        if ckpt.get('ema'):  #  if EMA is enabled, replace model with EMA
            ckpt['model'] = ckpt['ema']
        for k in ('optimizer', 'ema', 'updates'):  # remove optimizer, EMA, and updates
            ckpt[k] = None
        ckpt['epoch'] = epoch  # set the current epoch
        ckpt['model'].half()  # convert to FP16
        for p in ckpt['model'].parameters():  # disable gradients
            p.requires_grad = False
        torch.save(ckpt, ckpt_path)  # save the modified checkpoint
