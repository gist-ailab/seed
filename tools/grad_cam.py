# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import numpy as np
import os
from itertools import chain
import cv2
import tqdm
from PIL import Image


import detectron2.data.transforms as T
import torch
# from detectron2.config import get_cfg
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.detection_utils import read_image
# from detectron2.modeling import build_model
# from detectron2.data.datasets import register_coco_instances

from mask_eee_rcnn.data.detection_utils import read_image
from mask_eee_rcnn.modeling import build_model
from mask_eee_rcnn.checkpoint import DetectionCheckpointer

from mask_eee_rcnn.config import get_cfg
from mask_eee_rcnn.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from mask_eee_rcnn.data import detection_utils as utils
from mask_eee_rcnn.data.build import filter_images_with_few_keypoints
from mask_eee_rcnn.utils.logger import setup_logger
from mask_eee_rcnn.utils.visualizer import Visualizer
from mask_eee_rcnn.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch


import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAM():
    """
    Class to implement the GradCam function with it's necessary Pytorch hooks.

    Attributes
    ----------
    model : detectron2 GeneralizedRCNN Model
        A model using the detectron2 API for inferencing
    layer_name : str
        name of the convolutional layer to perform GradCAM with
    """

    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradient = None
        self.model.eval()
        self.activations_grads = []
        self._register_hook()
        self.model.zero_grad()

    def _get_activations_hook(self, module, input, output):
        self.activations = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.model.named_modules():
            if name == self.target_layer_name:
                self.activations_grads.append(module.register_forward_hook(self._get_activations_hook))
                self.activations_grads.append(module.register_backward_hook(self._get_grads_hook))
                return True
        print(f"Layer {self.target_layer_name} not found in Model!")

    def _release_activations_grads(self):
      for handle in self.activations_grads:
            handle.remove()
    
    def _postprocess_cam(self, raw_cam, img_width, img_height):
        cam_orig = np.sum(raw_cam, axis=0)  # [H,W]
        cam_orig = np.maximum(cam_orig, 0)  # ReLU
        cam_orig -= np.min(cam_orig)
        cam_orig /= np.max(cam_orig)
        cam = cv2.resize(cam_orig, (img_width, img_height))
        return cam, cam_orig
    
    def _forward_backward_pass(self, inputs, target_instance, target_pred):
        self.model.training = True
        output = self.model.forward([inputs])[0]
        if target_pred == "initial":
            ind = (output.pred_masks_initial[target_instance] >= 0.5)
            (output.pred_masks_initial[target_instance][ind]).sum().backward()
        elif target_pred == "refined":
            ind = (output.pred_masks[target_instance] >= 0.5)
            (output.pred_masks[target_instance][ind]).sum().backward()
        return output
    

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._release_activations_grads()

    def __call__(self, inputs, target_instance, target_pred):
        """
        Calls the GradCAM++ instance

        Parameters
        ----------
        inputs : dict
            The input in the standard detectron2 model input format
            https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format

        target_instance : int, optional
            The target category index. If `None` the highest scoring class will be selected

        Returns
        -------
        cam : np.array()
          Gradient weighted class activation map
        output : list
          list of Instance objects representing the detectron2 model output
        """

        output = self._forward_backward_pass(inputs, target_instance, target_pred)
        # gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        # activations = self.activations[0].cpu().data.numpy()  # [C,H,W]
        # weight = np.mean(gradient, axis=(1, 2))  # [C]

        # cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        # cam, cam_orig = self._postprocess_cam(cam, 28, 28)

        # gradcam ++
        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        activations = self.activations[0].cpu().data.numpy()  # [C,H,W]

        #from https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py
        grads_power_2 = gradient**2
        grads_power_3 = grads_power_2 * gradient
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(gradient != 0, aij, 0)

        weights = np.maximum(gradient, 0) * aij
        weight = np.sum(weights, axis=(1, 2))

        cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam, cam_orig = self._postprocess_cam(cam, 28, 28)

        return cam, cam_orig, output

class Detectron2GradCAM():
  """
      Attributes
    ----------
    config_file : str
        detectron2 model config file path
    cfg_list : list
        List of additional model configurations
    root_dir : str [optional]
        directory of coco.josn and dataset images for custom dataset registration
    custom_dataset : str [optional]
        Name of the custom dataset to register
    """
  def __init__(self, cfg, img_path, root_dir=None, custom_dataset=None):
      # load config from file

      if custom_dataset:
          register_coco_instances(custom_dataset, {}, root_dir + "coco.json", root_dir)
          cfg.DATASETS.TRAIN = (custom_dataset,)
          MetadataCatalog.get(custom_dataset)
          DatasetCatalog.get(custom_dataset)

      if torch.cuda.is_available():
          cfg.MODEL.DEVICE = "cuda"
      else:
          cfg.MODEL.DEVICE = "cpu"
    #   cfg.freeze()

      self.cfg =  cfg
      self._set_input_image(img_path)

  def _set_input_image(self, img_path):
      self.image = read_image(img_path, format="BGR")
      self.image_height, self.image_width = self.image.shape[:2]
      transform_gen = T.ResizeShortestEdge(
          [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
      )
      transformed_img = transform_gen.get_transform(self.image).apply_image(self.image)
      self.input_tensor = torch.as_tensor(transformed_img.astype("float32").transpose(2, 0, 1)).requires_grad_(True)
  
  def get_cam(self, target_instance, target_pred, layer_name, grad_cam_instance):
      """
      Calls the GradCAM instance

      Parameters
      ----------
      img : str
          Path to inference image
      target_instance : int
          The target instance index
      layer_name : str
          Convolutional layer to perform GradCAM on
      grad_cam_type : str
          GradCAM or GradCAM++ (for multiple instances of the same object, GradCAM++ can be favorable)

      Returns
      -------
      image_dict : dict
        {"image" : <image>, "cam" : <cam>, "output" : <output>, "label" : <label>}
        <image> original input image
        <cam> class activation map resized to original image shape
        <output> instances object generated by the model
        <label> label of the 
      cam_orig : numpy.ndarray
        unprocessed raw cam
      """
      model = build_model(self.cfg)
      print(model)
      checkpointer = DetectionCheckpointer(model)
      checkpointer.load(self.cfg.MODEL.WEIGHTS)

      input_image_dict = {"image": self.input_tensor, "height": self.image_height, "width": self.image_width}
      grad_cam = grad_cam_instance(model, layer_name)
    
      with grad_cam as cam:
        cam, cam_orig, output = cam(input_image_dict, target_instance=target_instance, target_pred=target_pred)
      
      output_dict = self.get_output_dict(cam, output, target_instance)
      
      return output_dict, cam_orig
    
  def get_output_dict(self, cam, output, target_instance):
      image_dict = {}
      image_dict["image"] =  self.image
      image_dict["cam"] = cam
      image_dict["output"] = output
    #   image_dict["label"] = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[output["instances"].pred_classes[target_instance]]
      return image_dict

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
     # automatically set output dir
    cfg.MODEL.WEIGHTS = args.config_file[:-5].replace("configs", "output") + "/model_final.pth"
    cfg.OUTPUT_DIR = args.config_file[:-5].replace("configs", "output")
    default_setup(cfg, args)
    return cfg



def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    # parser.add_argument(
    #     "--source",
    #     choices=["annotation", "dataloader"],
    #     required=True,
    #     help="visualize the annotations or the data loader (with pre-processing)",
    # )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    # parser.add_argument("--output-dir", default="./", help="path to output directory")
    # parser.add_argument("--show", action="store_true", help="show output in a window")
    # parser.add_argument(
    #     "opts",
    #     help="Modify config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger())
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    import glob 
    img_paths = glob.glob("/SSDc/sangbeom_lee/mask-eee-rcnn/datasets/cityscapes/leftImg8bit/val/lindau/*.png")
    for img_path in img_paths:
        # unfreeze the model

        cam_extractor = Detectron2GradCAM(cfg, img_path=img_path)
        grad_cam = GradCAM
        instance = 0
        target_pred = "initial"
        layer_name = "roi_heads.mask_head.mask_fcn3"
        image_dict, cam_orig = cam_extractor.get_cam(target_instance=instance, target_pred=target_pred, layer_name=layer_name, grad_cam_instance=grad_cam)


        rgb_crop = cv2.cvtColor(image_dict["image"], cv2.COLOR_BGR2RGB)
        # resize short edge to 800, maximum size to 1333
        transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        transformed_img = transform_gen.get_transform(rgb_crop).apply_image(rgb_crop)
        rgb_crop = transformed_img
        bbox = image_dict["output"].pred_boxes[instance].tensor[0].detach().cpu().numpy()
        bbox = bbox.astype(int)
        rgb_crop = rgb_crop[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        rgb_crop = cv2.resize(rgb_crop, (28, 28))
        plt.subplot(2, 3, 1)
        plt.imshow(rgb_crop)


        plt.subplot(2, 3, 2)
        plt.imshow(image_dict["output"].pred_masks_initial[instance].detach().cpu().numpy()[0])
        plt.subplot(2, 3, 3)
        plt.imshow(image_dict["cam"], cmap='jet', alpha=0.5)


        target_pred = "refined"
        layer_name = "roi_heads.mask_head.mask_fcn_refine3"
        image_dict, cam_orig = cam_extractor.get_cam(target_instance=instance, target_pred=target_pred, layer_name=layer_name, grad_cam_instance=grad_cam)
        plt.subplot(2, 3, 5)
        plt.imshow(image_dict["output"].pred_masks[instance].detach().cpu().numpy()[0])
        plt.subplot(2, 3, 6)
        plt.imshow(image_dict["cam"], cmap='jet', alpha=0.5)

        filename = os.path.basename(img_path)
        plt.savefig(f"grad_cam/{filename}_{instance}_cam.jpg")
        plt.tight_layout()
        plt.show()

        del cam_extractor
        # clean memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # clear plt
        plt.clf()




