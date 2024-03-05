import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from os import path as osp
from typing import List, Tuple

from tqdm import tqdm

VOC_NAMES = [
  'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
  'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


# pyright: reportArgumentType=false
def convert_label(path: str, label_path: str, year: str, image_id: str):
    """converts label from VOC format to YOLO format

    :param path: (str) the path to the directory containing VOC dataset
    :param label_path: (str) the path to the output label files in YOLO format
    :param year: (str) the year of the VOC dataset to use
    :param image_id: (str) the ID of the image
    """
    def convert_box(size: Tuple[int, int], box: List[float]) -> Tuple[float, ...]:
        """converts bounding box coordinates from absolute values to relative values

        :param size: (tuple) the image's resolution (in <width, height> format)
        :param box: (tuple) the bounding box coordinates (in <xmin, ymin, xmax, ymax> format)
        :return: a tuple containing the converted relative bounding box coordinates (in <x, y, w, h> format)
        """
        # calculate the reciprocal of the width and height of the image
        dw, dh = 1. / size[0], 1. / size[1]
        # calculate the center of the bounding box and its width and height
        x, y, w, h = (box[0] + box[2]) / 2.0 - 1, (box[1] + box[3]) / 2. - 1, box[2] - box[0], box[3] - box[1]
        # return the rescaled coordinates and dimensions
        return x * dw, y * dh, w * dw, h * dh

    # open the file containing annotations for the image
    in_file = open(osp.join(path, f'VOC{year}/Annotations/{image_id}.xml'))
    out_file = open(label_path, 'w')  # output file to store the converted labels

    tree = ET.parse(in_file)  # parse the XML file
    root = tree.getroot()  # get the root of XML tree
    size = root.find('size')  # find `size` element
    # extract width and height of the image
    w, h = int(size.find('width').text), int(size.find('height').text)
    for obj in root.iter('object'):  # iterate over all `object` elements in the XML tree
        cls = obj.find('name').text  # get the object's class name
        # if the class name is in the list of VOC names and the object is not marked as `difficult`
        if cls in VOC_NAMES and not int(obj.find('difficult').text) == 1:
            # find the `bndbox` element which contains the bounding box coordinates of the object
            xmlbox = obj.find('bndbox')
            # convert the bounding box coordinates to the format suitable for YOLO
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'ymin', 'xmax', 'ymax')])
            cls_id = VOC_NAMES.index(cls)  # get the class ID
            # write the class ID and the bounding box coordinates to the output file
            out_file.write(' '.join([str(a) for a in (cls_id, *bb)]) + '\n')

    # close file pointers
    in_file.close()
    out_file.close()


def generate_voc07_12(voc_path: str) -> None:
    """generate VOC07+12 setting dataset:
        train: # 16551 training images
            - images/train2012
            - images/train2007
            - images/val2012
            - images/val2007
        val: # 4952 val images (relative to 'path')
            - images/test2007

    :param voc_path: (str) the path to the VOC dataset
    """
    # define the root directory for the combined dataset
    dataset_root = osp.join(voc_path, 'voc_07_12')
    if not osp.exists(dataset_root):  # create if not exist
        os.makedirs(dataset_root, exist_ok=True)

    # define the settings for the combined dataset
    dataset_settings = {'train': ['train2007', 'val2007', 'train2012', 'val2012'], 'val': ['test2007']}
    for item in ['images', 'labels']:  # for each type of data (images and labels)
        for data_type, data_list in dataset_settings.items():  # iterate through types of set (train and val)
            for data_name in data_list:  # iterate through each dataset in the set
                # construct the original and new paths of the dataset
                original_path = osp.join(voc_path, item, data_name)
                new_path = osp.join(dataset_root, item, data_type)
                if not osp.exists(new_path):  # create combined dataset directory if not exist
                    os.makedirs(new_path, exist_ok=True)

                # logging and copying images
                print(f'[INFO]: Copying {original_path} to {new_path}')
                for file in os.listdir(original_path):
                    shutil.copy(osp.join(original_path, file), new_path)


def main(args: argparse.Namespace) -> None:
    """main function to convert VOC dataset to YOLO format

    :param args: (object) the command-line arguments
    """
    # get the VOC path from the command-line argument
    voc_path = args.voc_path
    # loop over different combinations of years and image sets
    for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
        # define the paths to the images and labels directories for the current image set
        image_paths = osp.join(voc_path, f'images/{image_set}{year}')
        label_paths = osp.join(voc_path, f'labels/{image_set}{year}')

        try:
            # open the file that contains the IDs of the images in the current image set
            with open(osp.join(voc_path, f'VOC{year}/ImageSets/Main/{image_set}.txt'), 'r') as f:
                image_ids = f.read().strip().split()  # clean and split

            # create if not exist
            if not osp.exists(image_paths):
                os.makedirs(image_paths, exist_ok=True)
            if not osp.exists(label_paths):
                os.makedirs(label_paths, exist_ok=True)

            for id in tqdm(image_ids, desc=f'{image_set}{year}'):  # iterate through the image IDs
                f = osp.join(voc_path, f'VOC{year}/JPEGImages/{id}.jpg')  # original image path
                label_path = osp.join(label_paths, f'{id}.txt')  # new label path
                convert_label(voc_path, label_path, year, id)  # convert labels to YOLO format
                if osp.exists(f):  # move image if exists
                    shutil.copy(f, image_paths)
        except Exception as e:  # if there was an error, log
            print(f'[WARNING]: {e} {year}{image_set} convert failed!')

    generate_voc07_12(voc_path)  # generate a combined VOC2007 and VOC2012 dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-path', type=str, default='VOCdevkit')

    args = parser.parse_args()
    print(args)

    main(args)
