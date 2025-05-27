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

logger = logging.getLogger(__name__)


def _get_rellis_files(image_dir, label_dir, ann_files):
    files = []
    
    for ann_file in ann_files:
        # CSV 파일을 읽어들입니다.
        with open(os.path.join(ann_file), 'r') as f:
            lines = f.readlines()

        # 각 라인에서 이미지 경로와 레이블 경로를 추출합니다.
        for line in lines:
            if '.png' not in line:  # png 확장자가 없는 라인은 무시합니다.
                continue
            line = line.strip()
            image_path, label_path = line.split(' ')[0], line.split(' ')[1]
            
            # 파일 경로에 대해 완전한 경로를 생성합니다.
            image_file = os.path.join(image_dir, image_path)
            label_file = os.path.join(label_dir, label_path)

            # 파일 경로를 튜플로 추가합니다.
            files.append((image_file, label_file))
    
    assert len(files), "No images found in the provided CSV files"
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_rellis_semantic_test(
    image_dir='/home/jovyan/SSDb/seongju_lee/dset/RELLIS/image',
    label_dir='/home/jovyan/SSDb/seongju_lee/dset/RELLIS/dgss_id',
    ann_files=['/home/jovyan/SSDb/seongju_lee/dset/RELLIS/split/test.lst'],
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
    for image_file, label_file in _get_rellis_files(image_dir, label_dir, ann_files):

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