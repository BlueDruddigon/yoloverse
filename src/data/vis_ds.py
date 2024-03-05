import argparse
import os

import cv2
import numpy as np

IMG_FORMATS = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'webp', 'mpo', 'dng']
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])


def main(args: argparse.Namespace) -> None:
    """visualizes the dataset by displaying images with bounding boxes and labels

    :param args: (object) the command-line arguments containing:
        - the path to the image directory
        - the path to the label directory
        - the path to the class names
    """
    # unpack the arguments
    image_dir, label_dir, class_names = args.image_dir, args.label_dir, args.class_names

    label_map = {}  # empty dictionary to map class IDs to class names
    for cls_id, class_name in enumerate(class_names):  # do the mappings
        label_map[cls_id] = class_name

    for file in os.listdir(image_dir):  # iterate over all files in the image directory
        if file.split('.')[-1] not in IMG_FORMATS:  # check the files' extension
            print(f'[WARNING]: Non-image file {file}')
            continue
        # construct the full path to the image and label files
        image_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, file[:file.rindex('.')] + '.txt')

        try:  # try processing the image
            image_data = cv2.imread(image_path)  # read the image by OpenCV
            height, width, _ = image_data.shape  # extract resolution
            # generate a random color for each class
            color = [tuple(np.random.choice(range(256), size=3)) for _ in class_names]
            thickness = 2

            with open(label_path, 'r') as f:  # open the label file and read the bounding box data
                for bbox in f:
                    # parse the bounding box data, converting the values to the appropriate types
                    cls, cx, cy, w, h = [
                      float(v) if i > 0 else int(v) for i, v in enumerate(bbox.split('\n')[0].split(' '))
                    ]
                    assert isinstance(cls, int)

                    # calculate the top-left corner of the bounding box
                    x_tl = int((cx - w/2) * width)
                    y_tl = int((cy - h/2) * height)

                    # draw the bounding box on the image
                    cv2.rectangle(
                      image_data, (x_tl, y_tl), (x_tl + int(w * width), y_tl + int(h * height)),
                      color=tuple([int(x) for x in color[cls]]),
                      thickness=thickness
                    )
                    # draw the class name above the bounding box
                    cv2.putText(
                      image_data,
                      label_map[cls], (x_tl, y_tl - 20),
                      cv2.FONT_HERSHEY_COMPLEX,
                      1,
                      color=tuple([int(x) for x in color[cls]]),
                      thickness=thickness
                    )

            # display the modified image with bounding box
            cv2.imshow('image', image_data)
            cv2.waitKey(0)
        except Exception as e:  # if there was an error, logging
            print(f'[ERROR]: {e} {image_path}')

    print('=====All Done!=====')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='VOCdevkit/voc_07_12/images')
    parser.add_argument('--label-dir', type=str, default='VOCdevkit/voc_07_12/labels')
    parser.add_argument(
      '--class-names',
      default=[
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
      ]
    )

    args = parser.parse_args()
    print(args)
