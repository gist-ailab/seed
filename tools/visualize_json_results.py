#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from mask_eee_rcnn.data import DatasetCatalog, MetadataCatalog
from mask_eee_rcnn.structures import Boxes, BoxMode, Instances
from mask_eee_rcnn.utils.logger import setup_logger
from mask_eee_rcnn.utils.visualizer import Visualizer
from mask_eee_rcnn.data.datasets import register_coco_instances


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])
    try:
        bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    except IndexError:
        bbox = []

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
        ret.pred_ee = [predictions[i]["ee"] for i in chosen]
        ret.pred_var = [predictions[i]["var"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    register_coco_instances('armbench_train', {}, 'datasets/armbench/mix-object-tote/train.json', 'datasets/armbench/mix-object-tote/images')
    register_coco_instances('armbench_val', {}, 'datasets/armbench/mix-object-tote/val.json', 'datasets/armbench/mix-object-tote/images')
    register_coco_instances('armbench_test', {}, 'datasets/armbench/mix-object-tote/test.json', 'datasets/armbench/mix-object-tote/images')

    register_coco_instances('armbench_train_class1', {}, 'datasets/armbench/mix-object-tote/train.json', 'datasets/armbench/mix-object-tote/images')
    register_coco_instances('armbench_val_class1', {}, 'datasets/armbench/mix-object-tote/val.json', 'datasets/armbench/mix-object-tote/images')
    register_coco_instances('armbench_test_class1', {}, 'datasets/armbench/mix-object-tote/test.json', 'datasets/armbench/mix-object-tote/images')

    register_coco_instances('cityscapes_val_cocofied', {}, 'datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json', 'datasets/cityscapes')


    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
