# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import json
import logging
import multiprocessing as mp
import numpy as np
import os
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

import glob

logger = logging.getLogger(__name__)


SPLITS = {
    'train': ['park-2', 'trail', 'trail-3', 'trail-4', 'trail-6', 'trail-9', 'trail-10', 'trail-11', 'trail-12', 'trail-14', 'trail-15', 'village'],
    'val': ['park-8', 'trail-5'],
    'test': ['creek', 'park-1', 'trail-7', 'trail-13']
}

def _get_rugd_files(image_dir, label_dir, split):
    files = []
    
    split_dirs = SPLITS.get(split, [])
    
    for split_dir in split_dirs:
        image_paths = sorted(glob.glob(os.path.join(image_dir, split_dir, '*.png')))
        label_paths = sorted(glob.glob(os.path.join(label_dir, split_dir, '*.png')))
        
        # 파일 경로를 튜플로 추가합니다.
        
        for image_file, label_file in zip(image_paths, label_paths):
            files.append((image_file, label_file))
    
    assert len(files), "No images found in the provided CSV files"
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_rugd_semantic_test(
    image_dir='/home/jovyan/SSDb/seongju_lee/dset/RUGD/frames',
    label_dir='/home/jovyan/SSDb/seongju_lee/dset/RUGD/dgss_id',
    split='test'
):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "dset/LaRS/lars_v1.0.0_images/train/images".
        gt_dir (str): path to the raw annotations. e.g., "dset/LaRS/lars_v1.0.0_annotations/train/semantic_masks_4cls".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    # gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, label_file in _get_rugd_files(image_dir, label_dir, split):
        # label_file = label_file.replace("labelIds", "labelTrainIds")

        # with PathManager.open(json_file, "r") as f:
        #     jsonobj = json.load(f)
        
        width, height = Image.open(label_file).size
        # print(height, width)
        # print("height, width", height, width)

        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": height,
                "width": width,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret

# def load_lars_semantic_val(image_dir='../dset/LaRS/lars_v1.0.0_images/val/images', gt_dir='../dset/LaRS/lars_v1.0.0_annotations/val/semantic_masks_4cls'):
#     """
#     Args:
#         image_dir (str): path to the raw dataset. e.g., "dset/LaRS/lars_v1.0.0_images/train/images".
#         gt_dir (str): path to the raw annotations. e.g., "dset/LaRS/lars_v1.0.0_annotations/train/semantic_masks_4cls".

#     Returns:
#         list[dict]: a list of dict, each has "file_name" and
#             "sem_seg_file_name".
#     """
#     ret = []
#     # gt_dir is small and contain many small files. make sense to fetch to local first
#     gt_dir = PathManager.get_local_path(gt_dir)
#     for image_file, label_file in _get_lars_files(image_dir, gt_dir):
#         label_file = label_file.replace("labelIds", "labelTrainIds")

#         # with PathManager.open(json_file, "r") as f:
#         #     jsonobj = json.load(f)
        
#         width, height = Image.open(label_file).size
#         # print(height, width)
#         # print("height, width", height, width)

#         ret.append(
#             {
#                 "file_name": image_file,
#                 "sem_seg_file_name": label_file,
#                 "height": height,
#                 "width": width,
#             }
#         )
#     assert len(ret), f"No images found in {image_dir}!"
#     assert PathManager.isfile(
#         ret[0]["sem_seg_file_name"]
#     ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
#     return ret