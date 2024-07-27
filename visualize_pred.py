
import torch
import sys
sys.path.insert(0, "Mask2Former")
import tempfile
from pathlib import Path
import numpy as np
import cv2
import random
import glob
# import some common detectron2 utilities
from mask_eee_rcnn.config import CfgNode as CN
from mask_eee_rcnn.engine import DefaultPredictor
from mask_eee_rcnn.config import get_cfg
from mask_eee_rcnn.utils.visualizer import CustomVisualizer, Visualizer, ColorMode
from mask_eee_rcnn.data import MetadataCatalog, DatasetCatalog
from pycocotools.coco import COCO
import os
from tqdm import tqdm
from mask_eee_rcnn.data.transforms import ResizeShortestEdge

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = CustomVisualizer(image, self.metadata, instance_mode=self.instance_mode)
        instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances, segmentation='pred_masks')
        
        initial_vis_output, error_vis_output = None, None
        if instances.has('pred_masks_initial'):
            initial_vis_output = visualizer.draw_instance_predictions(predictions=instances, segmentation='pred_masks_initial')
        if instances.has('pred_errors'):
            error_output = instances.pred_errors # N, H, W
            error_vis_output = image.copy()[:, :, ::-1] * 0.5
            for i in range(error_output.shape[0]):
                error_vis_output[error_output[i]] = (0, 0, 255) 
            if instances.has('pred_negative_errors'):
                error_negative_output = instances.pred_negative_errors # N, H, W
                for i in range(error_negative_output.shape[0]):
                    error_vis_output[error_negative_output[i]] = (0, 255, 0)


        return predictions, vis_output, initial_vis_output, error_vis_output



config_file_path = "configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs16_e3_re_l1_0.5_0.5_0.5.yaml"
dataset_name = 'coco_2017_val'
out_dir = 'vis'


cfg = get_cfg()
cfg.merge_from_file(config_file_path)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = config_file_path.replace('configs', 'output').replace('.yaml', '/model_final.pth')
demo = VisualizationDemo(cfg)
metadata = MetadataCatalog.get(dataset_name)
dataset_dicts = DatasetCatalog.get(dataset_name)

for idx, d in enumerate(tqdm(dataset_dicts)):
    img = cv2.imread(d["file_name"])
    
    min_size = cfg.INPUT.MIN_SIZE_TEST # size of the smallest size of the image
    max_size = cfg.INPUT.MAX_SIZE_TEST # max size of the side of the image
    tfm = ResizeShortestEdge(min_size, max_size).get_transform(img)
    resized = tfm.apply_image(img)

    predictions, pred_vis, pred_initial_vis, pred_error_vis = demo.run_on_image(resized)

    v = Visualizer(img[:, :, ::-1], metadata=metadata)
    gt_vis = v.draw_dataset_dict(d)

    pred_vis = pred_vis.get_image()[:, :, ::-1]
    pred_initial_vis = pred_initial_vis.get_image()[:, :, ::-1]
    gt_vis = gt_vis.get_image()[:, :, ::-1]
    gt_vis = cv2.resize(gt_vis, (pred_vis.shape[1], pred_vis.shape[0]))

    cv2.imwrite(out_dir + '/{}_{}.png'.format(os.path.basename(config_file_path)[:-5], idx), 
                np.vstack((
                    np.hstack((pred_initial_vis, pred_vis)),
                    np.hstack((pred_error_vis, gt_vis)))
                    ))


