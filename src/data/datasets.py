import glob
import hashlib
import json
import os
import os.path as osp
import random
import time
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import psutil
import torch
import torch.distributed as dist
from PIL import ExifTags, Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Self

from src.utils.logger import LOGGER

from .data_augment import augment_hsv, letterbox, mixup, mosaic_augmentation, random_affine

# GLOBAL PARAMs
IMG_FORMATS = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'webp', 'mpo', 'dng']
VID_FORMATS = ['mp4', 'mov', 'avi', 'mkv']
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])
CPU_COUNT = os.cpu_count()
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == 'Orientation':
        ORIENTATION = k
        break


def image2label_paths(image_paths: List[str]) -> List[str]:
    """transforms image paths to label paths by replace substrings

    :param image_paths: (list) a list containing all image paths
    :return: a list of corresponding label paths
    """
    # define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    # construct label paths by manipulating the provided image paths
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in image_paths]


class TrainValDataset(Dataset):
    def __init__(
      self,
      image_dir: str,
      img_size: int = 640,
      batch_size: int = 16,
      augment: bool = False,
      hyp: Optional[Dict[str, float]] = None,
      rect: bool = False,
      check_images: bool = False,
      check_labels: bool = False,
      stride: int = 32,
      pad: float = 0.0,
      rank: int = -1,
      data_dict: Optional[dict] = None,
      task: str = 'train',
      specific_shape: bool = False,
      height: int = 1080,
      width: int = 1920,
      cache_ram: bool = False,
      save_valid_record: bool = False
    ) -> None:
        """initializes the dataset object

        :param image_dir: (str) path to the directory containing images
        :param img_size: (int) target image size after augmentation. default: 640
        :param batch_size: (int) batch size for training. default: 16
        :param augment: (bool) whether applying augmentation. default: False
        :param hyp: (dict, optional) hyperparameters for augmentation. default: None
        :param rect: (bool) use rectangular targets for training. default: False
        :param check_images: (bool) check image integrity. default: False
        :param check_labels: (bool) check label validity. default: False
        :param stride: (int) stride for slicing window detector. default: 32
        :param pad: (float) padding ratio for letterboxing. default: 0.0
        :param rank: (int) rank of the process in distributed training. default: -1
        :param data_dict: (dict) dictionary containing data information. default: None
        :param task: (str) training task (train, val, test, or speed). default: 'train'
        :param specific_shape: (bool) whether using specific target shape for letterboxing. default: False
        :param height: (int) target height for specific shape. default: 1080
        :param width: (int) target width for specific shape. default: 1920
        :param cache_ram: (bool) whether caching images in RAM for faster access. default: False
        :param save_valid_record: (bool) whether saving validation records. default: False
        """
        super().__init__()
        # validate task name
        assert task.lower() in {'train', 'val', 'test', 'speed'}, f'task "{task}" is not supported'
        assert data_dict is not None  # validate `data_dict` presence
        start = time.time()  # recording time
        # update object attributes with input arguments
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1, 0)  # determine if the current process is the main process
        self.task = self.task.capitalize()  # capitalize task name
        self.class_names = data_dict['names']  # extract class names from `data_dict`

        # get image paths and labels
        self.image_paths, self.labels = self.get_images_labels(self.image_dir)
        # target resolution
        self.target_height = self.height
        self.target_width = self.width

        if self.rect:  # handle rectangular targets for training
            shapes = [self.image_info[p]['shape'] for p in self.image_paths]
            self.shapes = np.array(shapes, dtype=np.float64)

            if dist.is_initialized():  # handle distributed training for batch size and image indexing
                # in DDP mode, we need to make sure all images within `batch_size * num_gpus`
                # will resize and pad to the same shape
                sample_batch_size = self.batch_size * dist.get_world_size()
            else:
                sample_batch_size = self.batch_size
            # batch indices of each image
            self.batch_indices = np.floor(np.arange(len(shapes)) / sample_batch_size).astype(np.int_)
            self.sort_files_shapes()

        if self.cache_ram:  # cache images in RAM if enabled
            self.num_images = len(self.image_paths)
            # pre-allocate spaces
            self.images: List[Any] = [None] * self.num_images
            self.images_hw0: List[Any] = [None] * self.num_images
            self.images_hw: List[Any] = [None] * self.num_images
            self.cache_images(num_images=self.num_images)  # caching images

        if self.main_process:  # logging initialization time on the main process
            LOGGER.info(f'{time.time() - start:.1f}s for dataset initialization')

    def cache_images(self, num_images: Optional[int] = None) -> None:
        """caches a specified number of images in RAM for faster training

        :param num_images: (int, optional) the number of images to cache. if not provided, all images will be cached. default: None
        """
        # ensure `num_images` is specified
        assert num_images is not None, '`num_images` must be specified as '

        # check available memory
        mem = psutil.virtual_memory()  # get memory information
        mem_required = self.calculate_cache_occupy()  # calculate memory required for caching

        # disable caching if insufficient memory
        if mem_required > mem.available:
            self.cache_ram = False
            LOGGER.warning('Not enough RAM to cache images, caching is disabled.')
        else:
            gb = 1 << 30  # conversion factor for gigabytes

            # log memory information
            LOGGER.warning(
              f'{mem_required / gb:.1f}Gb RAM required, '
              f'{mem.available / gb:.1f}/{mem.total / gb:.1f}Gb RAM available, '
              'since the first thing we do is cache, '
              'there is no guarantee that the remaining memory space if sufficient'
            )

        # print and log information about caching
        print(f'self.images: {len(self.images)}')
        LOGGER.info(
          'You\'re using cached images in RAM to accelerate training!'
          'Caching images ...\n'
          'This might take some time for your dataset.'
        )

        # determine the number of threads to use
        assert CPU_COUNT is not None  # ensure CPU count is available
        num_threads = min(16, max(1, CPU_COUNT - 1))  # use up to 16 threads, but not more than CPUs-1

        # load images using threads
        load_images = ThreadPool(num_threads).imap(self.load_image, range(num_images))
        pbar = tqdm(enumerate(load_images), total=num_images, disable=self.rank > 0)

        # iterate through loaded images and update internal data structures
        for i, (x, (h0, w0), shape) in pbar:
            self.images[i], self.images_hw0[i], self.images_hw[i] = x, (h0, w0), shape

    def __del__(self) -> None:
        """ release the data in RAM if cached """
        if self.cache_ram:
            del self.images

    def __len__(self) -> int:
        """get the length of dataset

        :return: dataset's length
        """
        return len(self.image_paths)

    def __getitem__(
      self, index: int
    ) -> Tuple[Tensor, Tensor, str, Optional[Tuple[Tuple[int, int], Tuple[Tuple[float, float], Tuple[int, int]]]]]:
        """fetching a data sample for a given index.
        this function applies mosaic and mixup augmentations during training.
        during validation, letterbox augmentation is applied.

        :param index: (int) the index of the image to retrieve from the dataset
        :return: a tuple containing:
            - `image`: (Tensor) the preprocessed image in CHW format (RGB mode)
            - `labels_out`: (Tensor) the corresponding labels in a format suitable for the dataset (usually COCO format)
            - `image_path`: (str) the path to the image file
            - `shapes`: (tuple, optional) containing:
                1. original image shape before letterboxing
                2. target image size and padding information after letterboxing
                this information are set to None if using mosaic augmentation
        """
        # determine the target shape based on settings and batch information
        target_shape = ((self.target_height, self.target_width) if self.specific_shape else
                        self.batch_shapes[self.batch_indices[index]] if self.rect else self.img_size)

        if self.augment and random.random() < self.hyp['mosaic']:  # apply mosaic augmentation
            image, labels = self.get_mosaic(index, target_shape)
            shapes = None  # no shapes needed for mosaic augmentation

            if random.random() < self.hyp['mixup']:  # apply mixup augmentation
                image_other, labels_other = self.get_mosaic(random.randint(0, len(self.image_paths) - 1), target_shape)
                image, labels = mixup(image, labels, image_other, labels_other)
        else:
            # load the image and its original dimensions
            if self.hyp and 'shrink_size' in self.hyp:
                image, (h0, w0), (h, w) = self.load_image(index, self.hyp['shrink_size'])
            else:
                image, (h0, w0), (h, w) = self.load_image(index)

            # apply letterbox transformation for resizing and padding
            image, ratio, pad = letterbox(image, target_shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)  # for COCO mAP scaling

            # copy and update labels based on image dimensions and letterboxing
            labels = self.labels[index].copy()
            if labels.size:
                w *= ratio
                h *= ratio
                boxes = np.copy(labels[:, 1:])  # make a copy to avoid modifying the original labels
                boxes[:, 0] = w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]  # top-left x
                boxes[:, 1] = h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]  # top-left y
                boxes[:, 2] = w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]  # bottom-right x
                boxes[:, 3] = h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]  # bottom-right y
                labels[:, 1:] = boxes  # update bounding box coordinates to the corresponding labels

            if self.augment:  # apply random affine transformations for augmentation (if enabled)
                image, labels = random_affine(
                  image,
                  labels,
                  degrees=self.hyp['degrees'],
                  translate=self.hyp['translate'],
                  scale=self.hyp['scale'],
                  shear=self.hyp['shear'],
                  new_shape=target_shape
                )

        # clip labels to prevent out-of-bound values
        if len(labels):
            h, w = image.shape[:2]

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2 (clip to image boundaries)
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2 (clip to image boundaries)

            # convert bounding boxes to the normalized `xywh` format
            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # center x
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # center y
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            labels[:, 1:] = boxes  # update the bounding boxes to the corresponding labels

        if self.augment:  # apply additional general augmentations (if enabled)
            image, labels = self.general_augment(image, labels)

        # prepare output labels in Tensor
        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # convert from `HWC` to `CHW` and BGR to RGB formats
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)

        # return image and label tensors, the image path, and its shapes
        return torch.from_numpy(image), labels_out, self.image_paths[index], shapes

    def get_images_labels(self, image_dirs: Union[List[str], str]) -> Tuple[List[str], np.ndarray]:
        """gets image paths and corresponding labels, performing various checks and caching for efficiency

        :param image_dirs: (str | list) a list of image directory paths or single directory path
        :return: a tuple containing:
            - `image_paths`: (list) a list of valid image paths
            - `labels`: (list) a list of corresponding label data
        """
        if not isinstance(image_dirs, list):  # handle single directory input
            image_dirs = [image_dirs]
        # we store the cache image file in the first directory of `image_dirs`
        valid_image_record = osp.join(
          osp.dirname(image_dirs[0]),
          f'.{osp.basename(image_dirs[0])}_cache.json',
        )
        # set the number of worker threads for parallel image processing
        NUM_THREADS = min(8, CPU_COUNT if CPU_COUNT is not None else 8)

        # collect image paths from all directories
        image_paths = []
        for image_dir in image_dirs:
            assert osp.exists(image_dir), f'{image_dir} is an invalid directory path!'
            image_paths += glob.glob(osp.join(image_dir, '**/*'), recursive=True)

        # sort and filter image paths based on valid extensions
        image_paths = sorted(p for p in image_paths if p.split('.')[-1].lower() in IMG_FORMATS)

        # assert the presence of at least one image
        assert image_paths, f'No images found in {image_dirs[0]}'
        # generate a hash based on image paths
        image_hash = self.get_hash(image_paths)
        # log information about the cache file
        LOGGER.info(f'image record information path is: {valid_image_record}')

        # load image information from cache if available and valid
        if osp.exists(valid_image_record):
            with open(valid_image_record, 'r') as f:
                cache_info = json.load(f)
                if 'image_hash' in cache_info and cache_info['image_hash'] == image_hash:
                    image_info = cache_info['information']
                else:
                    self.check_images = True
        else:
            self.check_images = True

        # check image formats if needed and update cache
        if self.check_images and self.main_process:
            image_info = {}
            nc, msgs = 0, []  # number corrupt, messages
            # logging
            LOGGER.info(f'{self.task}: Checking formats of images with {NUM_THREADS} process(es): ')
            with Pool(NUM_THREADS) as pool:  # use a multiprocessing pool for efficient parallel processing
                # progress bar for visualization
                pbar = tqdm(pool.imap(TrainValDataset.check_image, image_paths), total=len(image_paths))
                # iterate over results from the pool
                for image_path, shape_per_image, nc_per_image, msg in pbar:
                    if nc_per_image == 0:  # store image information for valid images
                        image_info[image_path] = {'shape': shape_per_image}
                    nc += nc_per_image  # keep track of total corrupted images
                    if msg:  # collect any detailed error messages
                        msgs.append(msg)
                    pbar.desc = f'{nc} image(s) corrupted'  # update description

            pbar.close()  # close the progress bar
            if msgs:  # log any collected error messages
                LOGGER.info('\n'.join(msgs))

            # create cache information for potential storage
            cache_info = {'information': image_info, 'image_hash': image_hash}

            # save valid image paths
            if self.save_valid_record:
                with open(valid_image_record, 'w') as f:
                    json.dump(cache_info, f)

        # extract image paths and corresponding label paths
        image_paths = list(image_info.keys())
        label_paths = image2label_paths(image_paths)
        # assert the presence of label files
        assert label_paths, 'No labels found'
        # generate a hash based on label paths
        label_hash = self.get_hash(label_paths)
        # update the flag to check labels if label hash matched
        if 'label_hash' not in cache_info or cache_info['label_hash'] != label_hash:
            self.check_labels = True

        # check label formats if needed and update cache/image info
        if self.check_labels:
            cache_info['label_hash'] = label_hash
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number (missing, found, empty, corrupt) and messages
            # logging
            LOGGER.info(f'{self.task}: Checking formats of labels with {NUM_THREADS} process(es): ')
            with Pool(NUM_THREADS) as pool:  # utilize parallel processing
                # apply label checks across multiple processes
                pbar = pool.imap(TrainValDataset.check_label_files, zip(image_paths, label_paths))
                pbar = tqdm(pbar, total=len(label_paths))  # progress bar
                # iterate through results from the pool
                for (image_path, labels_per_file, nc_per_file, nm_per_file, nf_per_file, ne_per_file, msg) in pbar:
                    if nc_per_file == 0:  # store labels for the valid label files
                        image_info[image_path]['labels'] = labels_per_file
                    else:  # remove image entry if its label file is invalid
                        image_info.pop(image_path)

                    # update counters
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file

                    if msg:  # collect any detailed error messages
                        msgs.append(msg)

                    if self.main_process:  # update progress bar in the main process
                        pbar.desc = f'{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files'

            if self.main_process:  # actions performed only in the main process
                pbar.close()  # close the progress bar
                if self.save_valid_record:  # optionally save valid image information to cache
                    with open(valid_image_record, 'w') as f:
                        json.dump(cache_info, f)

            if msgs:  # log any collected error messages
                LOGGER.info('\n'.join(msgs))

            if nf == 0:  # issue a warning if no valid labels were found
                LOGGER.warning(f'WARNING: No labels found in {osp.dirname(image_paths[0])}')

        # handle COCO format evaluation if applicable
        if self.task.lower() == 'val':
            if self.data_dict.get('is_coco', False):  # use the original json file when evaluating on COCO dataset
                assert osp.exists(
                  self.data_dict['anno_path']
                ), 'Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml'
            else:
                assert self.class_names, 'Class names is required when converting labels to COCO format for evaluating'
                save_dir = osp.join(osp.dirname(osp.dirname(image_dirs[0])), 'annotations')
                if not osp.exists(save_dir):  # create directory if it doesn't exist
                    os.mkdir(save_dir)
                save_path = osp.join(save_dir, f'instances_{osp.basename(image_dirs[0])}.json')
                TrainValDataset.generate_coco_format_labels(image_info, self.class_names, save_path)

        # extract the final valid image paths and labels
        image_paths, labels = list(
          zip(
            *[(
              image_path,
              np.array(info['labels'], dtype=np.float32) if info['labels'] else np.zeros((0, 5), dtype=np.float32)
            ) for image_path, info in image_info.items()]
          )
        )
        # store image info and log final statistics
        self.image_info = image_info
        LOGGER.info(f'{self.task}: Final numbers of valid images: {len(image_paths)}, labels: {len(labels)}')

        return image_paths, labels

    def calculate_cache_occupy(self) -> float:
        """estimate the memory required to cache images in RAM

        :return: (float) the required memory
        """
        cache_bytes = 0  # store the accumulated memory usage
        num_images = len(self.image_paths)  # the total number of images in the dataset
        num_samples = min(num_images, 32)  # number of samples to use for estimation

        for _ in range(num_samples):  # loop through a random sample of images
            # load an image with random index
            img, _, _ = self.load_image(index=random.randint(0, num_images - 1))
            cache_bytes += img.nbytes  # extract and accumulate the number of bytes used by the image data

        # estimate the total memory required by scaling the sample-based usage
        return cache_bytes * num_images / num_samples

    def get_mosaic(self, index: int, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """creates a mosaic augmentation by combining four images into a single image

        :param index: (int) the index of the first image to use in the mosaic
        :param shape: (tuple) the desired output shape of the mosaic image
        :return: a tuple containing:
            - `image`: (np.ndarray) the resulting mosaic image
            - `labels`: (np.ndarray) the corresponding labels for the objects in the mosaic,
                formatted according to the dataset's requirements
        """
        # select additional image indices (excluding the first)
        indices = [index] + random.choices(range(len(self.image_paths)), k=3)  # 3 additional image indices
        random.shuffle(indices)  # randomly shuffle the indices

        images, hs, ws, labels = [], [], [], []  # initialize empty lists to store images, heights, widths, and labels
        # load each image and its corresponding labels
        for index in indices:
            image, _, (h, w) = self.load_image(index)
            labels_per_image = self.labels[index]
            # append to lists
            images.append(image)
            hs.append(h)
            ws.append(w)
            labels.append(labels_per_image)

        # apply the mosaic augmentation
        image, labels = mosaic_augmentation(
          shape, images, hs, ws, labels, self.hyp, self.specific_shape, self.target_height, self.target_width
        )
        return image, labels

    def general_augment(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """get images and labels after general augment
        this function applies hsv, random ud-flip and random lr-flip augments

        :param image: (np.ndarray) the image data
        :param labels: (np.ndarray) the ground-truth labels for corresponding image
        :return: a tuple containing:
            - (np.ndarray) the augmented image data
            - (np.ndarray) the corresponding augmented labels (if provided)
        """
        # extract the number of labels (if any)
        num_labels = len(labels)

        # apply HSV color-space augmentation
        augment_hsv(image, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])

        # apply random flip up-down augmentation
        if random.random() < self.hyp['flipud']:
            image = np.flipud(image)
            if num_labels:  # if labels exist, adjust bounding box coordinates by y-axis
                labels[:, 2] = 1 - labels[:, 2]

        # apply random flip left-right augmentation
        if random.random() < self.hyp['fliplr']:
            image = np.fliplr(image)
            if num_labels:  # if labels exist, adjust bounding box coordinates by x-axis
                labels[:, 1] = 1 - labels[:, 1]

        return image, labels

    def load_image(
      self,
      index: int,
      shrink_size: Optional[None] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, ...]]:
        """load image utility.
        this function loads image by cv2, resize the original image to the target shape with a keeping ratio

        :param index: (int) the index of image path
        :param shrink_size: (int) the shrink size for evaluation
        :return: a tuple of:
            - (np.ndarray) the loaded and potentially resized image
            - (tuple) original shape of image
            - (tuple) resized image's shape
        """
        # check for cached images
        if self.cache_ram and self.images[index] is not None:
            # return the cached image and its shapes
            return self.images[index], self.images_hw0[index], self.images_hw[index]

        # get the image path for given index
        path = self.image_paths[index]
        try:  # attempt to read image using OpenCV
            img = cv2.imread(path)
            # assert to ensure successful image reading
            assert img is not None, f'opencv cannot read image correctly or {path} is not exists!'
        except Exception as e:
            print(e)  # handle potential errors during OpenCV reading
            # fallback to Pillow for potentially corrupted images
            img = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
            # assert again to ensure successful loading after fallback
            assert img is not None, f'Image not found {path}, workdir: {os.getcwd()}'

        h0, w0 = img.shape[:2]  # original shape
        if self.specific_shape:  # if a specific target shape is defined
            # maintain the aspect ratio while resizing
            ratio = min(self.target_width / w0, self.target_height / h0)
        elif shrink_size:  # if `shrink_size` is provided (use for evaluation)
            ratio = (self.img_size - shrink_size) / max(h0, w0)
        else:  # default case (training mode)
            ratio = self.img_size / max(h0, w0)

        if ratio != 1:  # resize the image if the ratio is not 1 (original size)
            # choose interpolation method based on whether shrinking or not
            img = cv2.resize(
              img, (int(w0 * ratio), int(h0 * ratio)),
              interpolation=cv2.INTER_AREA if ratio < 1 and not self.augment else cv2.INTER_LINEAR
            )

        return img, (h0, w0), img.shape[:2]

    @staticmethod
    def collate_fn(
      batch: List[Tuple[Tensor, Tensor, str, Tuple[int, int]]]
    ) -> Tuple[Tensor, Tensor, str, Tuple[int, int]]:
        """merges a list of samples to form a mini-batch of Tensors

        :param batch: (list) a list of data samples, where each sample is a tuple containing:
            - `image` (Tensor): the image data
            - `label` (Tensor): the target labels associated with the image
            - `path` (str): the path to the image file
            - `shapes` (tuple) the original image shape before any transformations
        :return: a tuple containing the modified information above
        """
        # unpack elements from each sample in the batch
        image, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for `build_targets()`
        # combine elements into a mini-batch of Tensors
        return torch.stack(image, dim=0), torch.cat(label, dim=0), path, shapes

    def sort_files_shapes(self) -> None:
        """ sort by aspect ratio """
        # get the number of samples in the last batch
        batch_num = self.batch_indices[-1] + 1
        s = self.shapes  # (H, W)
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()  # sort indices based on an ascending aspect ratio

        # reorder image paths, labels, and shapes according to the sorted indices
        self.image_paths = [self.image_paths[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = s[irect]  # reordered shapes
        ar = ar[irect]  # reordered aspect ratios

        # set training image shapes
        shapes = [[1, 1]] * batch_num
        for i in range(batch_num):  # calculate target shapes for each batch based on aspect ratios
            ari = ar[self.batch_indices == i]  # aspect ratios within the current batch
            mini, maxi = ari.min(), ari.max()
            # define target shape based on aspect ratio range
            if maxi < 1:  # wide images (width < height)
                shapes[i] = [1, maxi]
            elif mini > 1:  # tall images (width > height)
                shapes[i] = [1 / mini, 1]

        # calculate final batch shapes considering image size, stride, and padding
        self.batch_shapes = (
          np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(np.int_) * self.stride
        )

    @staticmethod
    def check_image(image_file: str) -> Tuple[str, Optional[Tuple[int, int]], int, str]:
        """verify an image

        :param image_file: (str) path of the image
        :return: a tuple contains:
            - `image_file`: (str) the original path of image
            - `shape`: (tuple, optional) the image's resolution
            - `nc`: (int) representing for an invalid image. 0 for valid image, 1 for invalid
            - `msg`: (str) the checking messages
        """
        nc, msg = 0, ''  # initialize variables to track validity and messages
        try:  # try opening the image using Pillow
            image = Image.open(image_file)
            image.verify()  # built-in PIL verify
            image = Image.open(image_file)  # need to reload the image after using `verify()`
            shape = (image.height, image.width)
            try:  # check for potential EXIF orientation information and adjust shape if needed
                image_exif = image._getexif()
                if image_exif and ORIENTATION in image_exif:
                    rotation = image_exif[ORIENTATION]
                    if rotation in (6, 8):
                        shape = (shape[1], shape[0])  # swap width and height for specific rotations
            except Exception:
                image_exif = None

            assert shape[0] > 9 and shape[1] > 9, f'image size {shape} < 10 pixels'
            assert image.format.lower() in IMG_FORMATS, f'invalid image format {image.format}'

            # special handling for JPEG/JPG images
            if image.format.lower() in ('jpg', 'jpeg'):
                # check for corrupted JPEG files (missing end marker)
                with open(image_file, 'rb') as f:
                    f.seek(-2, 2)  # seek to the second-last byte
                    if f.read() != b'\xff\xd9':  # check for expected JPEG end marker
                        # attempt to repair the image by reading, transposing (if needed), and saving
                        image = Image.open(image_file)
                        ImageOps.exif_transpose(image).save(image_file, 'JPEG', subsampling=0, quality=100)
                        msg += f'WARNING: {image_file}: corrupt JPEG restored and saved'

            return image_file, shape, nc, msg
        except Exception as e:  # handle any exceptions during image processing
            nc = 1  # set validity flag to 1 (invalid)
            msg = f'WARNING: {image_file}: ignoring corrupted image: {e}'
            return image_file, None, nc, msg

    @staticmethod
    def check_label_files(args: Tuple[str, str]) -> Tuple[str, Optional[List[float]], int, int, int, int, str]:
        """verify a label file

        :param args: (tuple) containing the image and label path
        :return: a tuple containing:
            - `image_path`: (str) original image path
            - `labels`: (optional) a list of ground-truth objects with verified format
            - `nc`: (int) representing invalid labels. default: 0 (1 for invalid file)
            - `nm`: (int) representing missing labels. default: 0 (1 for missing)
            - `nf`: (int) representing labels found. default: 0 (1 for labels found)
            - `ne`: (int) representing an empty file. default: 0 (1 for empty file)
            - `msg`: (str) the checking messages
        """
        image_path, label_path = args  # extract image and label paths from the input tuple
        # initialize counters and message string
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ''  # num (missing, found, empty, invalid), messages
        try:
            if osp.exists(label_path):  # check if the label file exists
                nf = 1  # labels found

                # open and read the label file content
                with open(label_path, 'r') as f:
                    # read, clean lines and split lines into label entries
                    labels = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    labels = np.array(labels, dtype=np.float32)  # convert to NumPy array

                if len(labels):  # check if any labels were found
                    # correct format assertion
                    assert all(len(lb) == 5 for lb in labels), f'{label_path}: wrong label format.'
                    assert (labels >= 0).all(), f'{label_path}: label values error: all values in label file must > 0'
                    assert (labels[:, 1:] <= 1).all(), \
                        f'{label_path}: label values error: all coordinates must be normalized'

                    # check for duplicate labels and remove them if found
                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        msg += f'WARNING: {label_path}: {len(labels) - len(indices)} duplicate labels removed'
                    labels = labels.tolist()  # convert labels back to list
                else:
                    ne = 1  # empty labels
                    labels = []
            else:
                nm = 1  # missing labels
                labels = []

            return image_path, labels, nc, nm, nf, ne, msg
        except Exception as e:  # handle any exceptions during label file processing
            nc = 1  # set validity flag to 1 (invalid)
            msg = f'WARNING: {label_path}: ignoring invalid labels: {e}'
            return image_path, None, nc, nm, nf, ne, msg

    @staticmethod
    def generate_coco_format_labels(image_info: dict, class_names: List[str], save_path: str):
        """generates COCO format annotations from a given image information dict

        :param image_info: (dict) containing information for each image, including:
            - image path
            - image shape (height, width)
            - labels (optional, list of bounding boxes in [cls_id, x, y, w, h] format)
        :param class_names: (list) a list of class names corresponding to the class IDs in the labels
        :param save_path: (str) the path to save the COCO format annotations in JSON format
        """
        # for evaluation with pycocotools
        dataset = {'categories': [], 'annotations': [], 'images': []}  # initialize an empty COCO annotation structure
        # create COCO category definitions (one for each class)
        for i, class_name in enumerate(class_names):
            dataset['categories'].append({'id': i, 'name': class_name, 'supercategory': ''})

        annotation_id = 0
        LOGGER.info('Convert to COCO format')

        # iterate through each image and its information
        for image_path, info in tqdm(image_info.items()):
            # extract image ID, height, and width from the information
            image_id = osp.splitext(osp.basename(image_path))[0]
            image_height, image_width = info['shape']

            # create COCO image annotation
            dataset['images'].append({
              'file_name': osp.basename(image_path),
              'id': image_id,
              'width': image_width,
              'height': image_height
            })

            if labels := info['labels'] or []:
                for label in labels:
                    # extract class id, center coordinates, and width/height
                    c, x, y, w, h = label[:5]

                    # convert the bounding box format from `xywh` to `xyxy`
                    x1 = (x - w/2) * image_width
                    y1 = (y - h/2) * image_height
                    x2 = (x + w/2) * image_width
                    y2 = (y + h/2) * image_height

                    # convert class id to zero-based indexing (COCO starts from 0)
                    cls_id = int(c)
                    # ensure valid width and height (avoid negative values)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)

                    # create COCO annotation for each bounding box
                    dataset['annotations'].append({
                      'area': h * w,  # area of the bounding box
                      'bbox': [x1, y1, x2, y2],  # bounding box coordinates
                      'category_id': cls_id,  # class id (zero-based)
                      'id': annotation_id,  # unique annotation id
                      'image_id': image_id,  # id of the image containing the bounding box
                      'iscrowd': 0,  # not a crowd instance (default)
                      'segmentation': [],  # empty mask (not used in COCO detection)
                    })
                    annotation_id += 1  # increment annotation id counter

        # save the COCO annotations dictionary to a JSON file
        with open(save_path, 'w') as f:
            json.dump(dataset, f)
        # logging message
        LOGGER.info(f'Convert to COCO format finished. Results saved in {save_path}')

    @staticmethod
    def get_hash(paths: List[str]) -> str:
        """get the hash value of paths

        :param paths: (List[str]) list of image paths
        :return: hash value of paths
        """
        assert isinstance(paths, list), 'Only support list currently'
        # combine all paths to single string and encode it
        h = hashlib.md5(''.join(paths).encode())  # using MD5 hash object with encoded string
        return h.hexdigest()  # heximal representation of calculated hash value


class LoadData:
    def __init__(self, path: str, webcam: bool, webcam_addr: str) -> None:
        """initializes the data source, handling webcam or file-based input

        :param path: (str) path to files or directory containing images or videos, or a single file path
        :param webcam: (bool) whether using a webcam for input
        :param webcam_addr: (str) address of the webcam device if `webcam` is True
        :raises FileNotFoundError: Invalid path
        """
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        if webcam:  # using webcam input
            image_path = []  # no image paths for webcam
            video_path = [int(webcam_addr) if webcam_addr.isdigit() else webcam_addr]  # access webcam using its address
        else:  # handling file-based input
            path = str(Path(path).resolve())  # ensure absolute path for consistent handling
            if osp.isdir(path):  # gather all files recursively
                files = sorted(glob.glob(osp.join(path, '**/*.*'), recursive=True))
            elif osp.isfile(path):  # use the provided path
                files = [path]  # files
            else:  # raise error if invalid path
                raise FileNotFoundError(f'Invalid path {path}')

            # separate image and video paths
            image_path = [i for i in files if i.split('.')[-1] in IMG_FORMATS]
            video_path = [v for v in files if v.split('.')[-1] in VID_FORMATS]

        self.files = image_path + video_path  # full list of files to process
        self.num_files = len(self.files)  # total number of files
        self.type = 'image'  # default type is image
        if video_path:  # video present: open the first video
            self.add_video(video_path[0])  # new video
        else:  # no video capture needed
            self.cap = None

    def checkext(self, path: Union[str, int]) -> str:
        """determines the type of data source (image or video) based on input path and webcam flag

        :param path: (str | int) path to the file or the webcam address
        :return: (str) the type of data source ('video' or 'image')
        """
        if self.webcam:  # webcam is always considered as video
            return 'video'
        else:  # for files, check the file extension against image formats
            return 'image' if path.split('.')[-1].lower() in IMG_FORMATS else 'video'

    def __iter__(self) -> Self:
        """returns an iterator over the class object.
        this method is called when this class object is used in a `for` loop.
        it resets the internal counter and returns itself as the iterator.

        :return: self
        """
        self.count = 0  # reset the counter
        return self

    def __next__(self) -> Tuple[np.ndarray, Union[str, int], Optional[cv2.VideoCapture]]:
        """retrieves the next item from the dataset, handling both images and videos.
        this method is called when iterating over the dataset using a `for` loop.
        it fetches the next file from the list, checks its type, and returns the corresponding data (image or video frame).

        :return: a tuple containing:
            - (np.ndarray) the next image or video frame
            - (str | int) the path of the retrieved file
            - (cv2.VideoCapture) the OpenCV video capture object if reading from a video or webcam, otherwise None
        :raises StopIteration:
        """
        if self.count == self.num_files:  # no more files
            raise StopIteration
        path = self.files[self.count]  # get the current data source
        if self.checkext(path) == 'video':  # check if type of current path is video
            self.type = 'video'  # update type
            ret_val, image = self.cap.read()  # read frame
            while not ret_val:
                self.count += 1  # increment the counter
                self.cap.release()  # close previous video
                if self.count == self.num_files:  # raise error when reaching last video
                    raise StopIteration
                path = self.files[self.count]  # get next data source
                self.add_video(path)  # open capture for the new video
                ret_val, image = self.cap.read()  # read frame from the new video
        else:  # read image
            self.count += 1  # increment the counter
            assert isinstance(path, str), f'Image path {path} must be string'
            image = cv2.imread(path)  # BGR

        return image, path, self.cap

    def add_video(self, path: Union[str, int]) -> None:
        """initializes a video capture for a new video

        :param path: (str | int) path to the video file or webcam address
        """
        self.frame = 0  # reset frame counter for new video
        self.cap = cv2.VideoCapture(path)  # open video capture using OpenCV

        if self.cap.isOpened():  # check if video capture opened successfully
            # get total number of frames in the video (if supported)
            self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:  # handle potential errors opening the video
            print(f'Error opening video: {path}')

    def __len__(self) -> int:
        """retrieve the length of this dataset

        :return: (int) the class object length
        """
        return self.num_files
