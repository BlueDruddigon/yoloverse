import math
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np


def augment_hsv(image: np.ndarray, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
    """HSV color-space augmentation

    :param image: (np.ndarray) the input image in BGR format
    :param hgain: (float) gain factor for Hue adjustment. default: 0.5
    :param sgain: (float) gain factor for Saturation adjustment. default: 0.5
    :param vgain: (float) gain factor for Value adjustment. default: 0.5
    """
    if hgain or sgain or vgain:  # check non-zero values
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        # apply BGR to HSV conversion and split channels
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # `np.uint8`

        # generate Lookup Table (LUT)
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 100).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # apply channel-wise LUT augmentation and merge into HSV then convert back to BGR
        image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def letterbox(
  image: np.ndarray,
  new_shape: int | List[int] | Tuple[int, int] = (640, 640),
  color: Tuple[int, int, int] = (114, 114, 114),
  auto: bool = True,
  scaleup: bool = True,
  stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resizes and pads an image to a desired `new_shape` while maintaining aspect ratio.

    :param image: (np.ndarray) the input image
    :param new_shape: (int | tuple) the desired new shape for image. default: (640, 640)
    :param color: (tuple) the color to use for padding the image (BGR format). default: (114, 114, 114)
    :param auto: (bool) whether pad the image with minimum rectangle to meet the new shape while
        maintaining aspect ratio, aligning the padding to the `stride`.
        if False, pad with equal amounts on each side. default: True
    :param scaleup: (bool) whether allows scaling image up (scale down if False). default: True
    :param stride: (int) the stride for padding alignment (only if `auto` is True). default: 32
    :return: a tuple containing:
        - the resized and padded image
        - the scaling ratio applied to the image
        - a tuple representing the padding applied to the left and top sides of the image
    """
    shape = image.shape[:2]  # current shape
    # handle different input formats for `new_shape`
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, (list, tuple)) and len(new_shape) == 1:
        new_shape = (new_shape[0], new_shape[0])

    # calculate scale ratio based on the minimum required scaling factor to fit the new shape
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        ratio = min(ratio, 1.0)

    # compute padding
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[0]  # wh padding

    if auto:  # minimum rectangle
        # ensure padding is a multiple of the stride
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize if necessary
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # calculate padding amounts for top, bottom, left, right sides
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # add padding to the image
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return image, ratio, (left, top)


def mixup(
  image: np.ndarray,
  labels: np.ndarray,
  image2: np.ndarray,
  labels2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf

    :param image: (np.ndarray)  the first image data
    :param labels: (np.ndarray) labels for the first image
    :param image2: (np.ndarray) the second image data
    :param labels2: (np.ndarray) labels for the second image
    :return: a tuple containing the mixed image and the concatenated labels
    """
    # generate a random mixup ratio based on the distribution
    ratio = np.random.beta(32.0, 32.0)  # mixup ratio, alpha = beta = 32.0

    # mix the two images using the generated mixup ratio
    image = (image*ratio + image2 * (1-ratio)).astype(np.uint8)

    # concatenate the labels of the two images along axis 0
    labels = np.concatenate((labels, labels2), axis=0)
    return image, labels


def box_candidates(
  box1: np.ndarray,
  box2: np.ndarray,
  wh_thres: int = 2,
  ar_thres: float = 20,
  area_thres: float = 0.1,
  eps: float = 1e-16
) -> bool:
    """evaluate if a box after augmentation (box2) is a valid candidate
    based on its dimensions and changes compared to the original box (box1)

    :param box1: (np.ndarray) the coordinates of the original bounding box (xyxy format)
    :param box2: (np.ndarray) the coordinates of the augmented bounding box (xyxy format)
    :param wh_thres: (int) minimum width and height threshold (pixels). default: 2
    :param ar_thres: (float) aspect ratio threshold. default: 20
    :param area_thres: (float) minimum area ratio threshold relative to the original box area. default: 0.1
    :param eps: (float) small value to avoid division by zero. default: 1e-16
    :return: (bool) whether the augmented box is considered a valid candidate.
    """
    # extract width and height for both boxes
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    # calculate aspect ratio of the augmented box (box2)
    ar = np.maximum(w2 / (h2+eps), h2 / (w2+eps))

    # define conditions for considering a box as candidate:
    # 1. minimum width and height threshold
    # 2. aspect ratio within the specified threshold
    # 3. area after augmentation is a significant portion of the original area
    return (w2 > wh_thres) & (h2 > wh_thres) & (w2 * h2 / (w1*h1 + eps) > area_thres) & (ar < ar_thres)


def random_affine(
  image: np.ndarray,
  labels: np.ndarray,
  degrees: float = 10,
  translate: float = 0.1,
  scale: float = 0.1,
  shear: float = 10,
  new_shape: int | Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, np.ndarray]:
    """applies random affine transformations to an image and its bounding box labels

    :param image: (np.ndarray) the image to be transformed
    :param labels: (np.ndarray, optional) bounding box labels for the objects in the image
    :param degrees: (float) maximum degree of rotation (positive or negative) to be applied randomly. default: 10
    :param translate: (float) maximum translation distance (as a proportion of the image size) to be applied randomly. default: 0.1
    :param scale: (float) maximum scaling factor (as a proportion of the original size) to be applied randomly. default: 0.1
    :param shear: (float) maximum shear angle (in degrees) to be applied randomly along both x and y axes. default: 10
    :param new_shape: (int | tuple) the desired output shape of the transformed image. default: (640, 640)
    :return: a tuple containing:
        - (np.ndarray) the transformed image
        - (np.ndarray) the transformed bounding box labels
    """
    n = len(labels)  # number of bounding boxes (if `labels` are provided)
    # handle single value for `new_shape` (assuming equal height and width)
    if isinstance(new_shape, int):
        height = width = new_shape
    else:
        height, width = new_shape

    # get the transformation matrix and scaling factor
    M, s = get_transform_matrix(image.shape[:2], (height, width), degrees, scale, shear, translate)
    if (M != np.eye(3)).any():  # applying transformations if image changed
        image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # transform label coordinates
    if n:
        new = np.zeros((n, 4))  # initialize array to store transformed bounding boxes

        # reshape labels and create a point array with repeated coordinates for transformations
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T  # apply transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # calculate bounding box coordinates from transformed points
        x = xy[:, [0, 2, 4, 6]]  # x-coordinates of top-left, bottom-left, top-right, bottom-right
        y = xy[:, [1, 3, 5, 7]]  # y-coordinates of top-left, bottom-left, top-right, bottom-right
        # combine into new bounding box format
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip coordinates to the image boundaries
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter coordinates based on area threshold
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thres=0.1)
        labels = labels[i]  # keep valid bounding boxes
        labels[:, 1:5] = new[i]  # update labels with the transformed bounding boxes

    return image, labels


def get_transform_matrix(
  image_shape: Tuple[int, ...], new_shape: Tuple[int, int], degrees: float, scale: float, shear: float, translate: float
) -> Tuple[np.ndarray, float]:
    """generate a 3x3 transformation matrix for applying geometric transformations to an image

    :param image_shape: (tuple) the original image's resolution
    :param new_shape: (tuple) the desired new resolution of the transformed image
    :param degrees: (float) maximum degree of rotation (positive or negative) to be applied randomly
    :param scale: (float) maximum scaling factor (as a proportion of original size) to be applied randomly
    :param shear: (float) maximum shear angle (in degrees) to be applied randomly along both x and y axes
    :param translate: (float) maximum translation distance (as a proportion of the new image size) to be applied randomly along both x and y axes
    :return: a tuple containing:
        - (np.ndarray) the 3x3 transformation matrix
        - the randomly generated scaling factor
    """
    assert len(image_shape) == 2
    new_height, new_width = new_shape
    # center
    C = np.eye(3)
    C[0, 2] = -image_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image_shape[0] / 2  # y translation (pixels)

    # rotation and scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)  # random rotation angle
    s = random.uniform(1 - scale, 1 + scale)  # random scaling factor
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y translation (pixels)

    # combined rotation matrix
    M = T @ S @ R @ C  # NOTE: order of operations (right to left) is `IMPORTANT`
    return M, s


def mosaic_augmentation(
  shape: int | Tuple[int, int],
  images: List[np.ndarray],
  hs: List[int],
  ws: List[int],
  labels: List[np.ndarray],
  hyp: Dict[str, float],
  specific_shape: bool = False,
  target_height: int = 640,
  target_width: int = 640
) -> Tuple[np.ndarray, np.ndarray]:
    """implements Mosaic Augmentation for image and label data during training

    :param shape: (int | tuple) the desired shape of the output image
    :param images: (list) a list of images to be combined
    :param hs: (list) a list of image heights
    :param ws: (list) a list of image widths
    :param labels: (list) a list of bounding box labels for each image
    :param hyp: (dict) a dictionary containing hyperparameter values for random affine transformations
    :param specific_shape: (bool) whether the provided shape is specific (True)
        or should be treated as target height and width (False). default: False
    :param target_height: (int) target height for the output image. default: 640
    :param target_width: (int) target width for the output image. default: 640
    :return: a tuple containing:
        - (np.ndarray) the augmented image
        - (np.ndarray) the augmented bounding box labels
    """
    # ensure 4 images for processing
    assert len(images) == 4, 'Mosaic Augmentation of current version only supports 4 images'
    labels4 = []  # storing augmented labels
    if not specific_shape:  # handle different input shape formats
        if isinstance(shape, (list, tuple, np.ndarray)):
            target_height, target_width = shape
        else:
            target_height = target_width = shape

    # generate random mosaic center coordinates
    yc, xc = (int(random.uniform(x // 2, 3 * x // 2)) for x in (target_height, target_width))

    for i, (image, h, w) in enumerate(zip(images, hs, ws)):  # loop through each image
        # extract image data and calculate coordinates for placing it within the mosaic
        if i == 0:  # top-left
            # initialize mosaic image (filled with gray)
            image4 = np.full((target_height * 2, target_width * 2, image.shape[2]), 114, dtype=np.uint8)
            # coordinates (xyxy format) for large image placement
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            # coordinates (xyxy format) for small image placement
            x1b, y1b, x2b, y2b = w - (x2a-x1a), h - (y2a-y1a), w, h
        elif i == 1:  # top-right
            # coordinates (xyxy format) for large image placement
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_width * 2), yc
            # coordinates (xyxy format) for small image placement
            x1b, y1b, x2b, y2b = 0, h - (y2a-y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            # coordinates (xyxy format) for large image placement
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_height * 2, yc + h)
            # coordinates (xyxy format) for small image placement
            x1b, y1b, x2b, y2b = w - (x2a-x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom-right
            # coordinates (xyxy format) for large image placement
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_width * 2), min(yc + h, target_height * 2)
            # coordinates (xyxy format) for small image placement
            x1b, y1b, x2b, y2b = 0, 0, min(x2a - x1a, w), min(y2a - y1a, h)
        else:
            raise ValueError

        # copy the image data into the corresponding section of the mosaic image
        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        # calculate padding amounts based on the placement coordinates
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        # handle bounding box labels if provided
        labels_per_image = labels[i].copy()  # copy labels for current image
        if labels_per_image.size:  # check if there are any labels for this image
            boxes = np.copy(labels_per_image[:, 1:])  # extract bounding box coordinates
            boxes[:, 0] = w * (labels_per_image[:, 1] - labels_per_image[:, 3] / 2) + pad_w  # top left x
            boxes[:, 1] = h * (labels_per_image[:, 2] - labels_per_image[:, 4] / 2) + pad_h  # top left y
            boxes[:, 2] = w * (labels_per_image[:, 1] + labels_per_image[:, 3] / 2) + pad_w  # bottom right x
            boxes[:, 3] = h * (labels_per_image[:, 2] + labels_per_image[:, 4] / 2) + pad_h  # bottom right y
            # update the bounding box coordinates in the labels
            labels_per_image[:, 1:] = boxes

        labels4.append(labels_per_image)  # append the modified labels for this image

    # concatenate labels from all images into a single array
    labels4 = np.concatenate(labels4, axis=0)
    # clip bounding box coordinates to be within the image boundaries (0 - 2x target dimensions)
    labels4[:, 1::2] = np.clip(labels4[:, 1::2], 0, 2 * target_width)
    labels4[:, 2::2] = np.clip(labels4[:, 2::2], 0, 2 * target_height)

    # apply random affine transformations to the combined image and labels (if provided)
    image4, labels4 = random_affine(
      image4,
      labels4,
      degrees=hyp['degrees'],
      translate=hyp['translate'],
      scale=hyp['scale'],
      shear=hyp['shear'],
      new_shape=(target_height, target_width)
    )

    return image4, labels4
