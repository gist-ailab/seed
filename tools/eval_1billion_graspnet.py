import os
import cv2
import glob
import numpy as np
import imageio
import torch
import json

from tqdm import tqdm
from termcolor import colored

# from detectron2.engine import DefaultPredictor
# from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

# from detectron2.config import get_cfg
# from detectron2.data.detection_utils import annotations_to_instances
# from detectron2.data.datasets.coco import load_coco_json

import mask_eee_rcnn.utils.comm as comm
from mask_eee_rcnn.checkpoint import DetectionCheckpointer
from mask_eee_rcnn.config import get_cfg
from mask_eee_rcnn.data import MetadataCatalog
from mask_eee_rcnn.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, DefaultPredictor
from mask_eee_rcnn.evaluation import (
    SemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
from detectron2.evaluation import CityscapesInstanceEvaluator
from mask_eee_rcnn.modeling import GeneralizedRCNNWithTTA
from mask_eee_rcnn.data.datasets import register_coco_instances
from mask_eee_rcnn.data.datasets.coco import load_coco_json

import compute_PRF
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        elif evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        # elif evaluator_type == "pascal_voc":
        #     return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.OUTPUT_DIR = args.config_file[:-5].replace("configs", "output")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    
    predictor = DefaultPredictor(cfg)
    # model = Trainer.build_model(cfg)
    data_root_path = 'datasets/1billion_graspnet'
    # W, H = cfg.INPUT.IMG_SIZE
    
    # load dataset
    json_path = 'datasets/1billion_graspnet/annotations/realsense/test_seen.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    images = data['images']
    
    # coco_json = load_coco_json(json_path, image_root=data_root_path)
    # annos = data['annotations']
    # targets = annotations_to_instances(annos, (720, 1280))
    # print(targets)

    rgb_paths = []
    metrics_all = []
    num_inst_mat = 0
    iou_masks = 0
    for i in range(len(images)):
        rgb_paths.append(images[i]['file_name'])
    rgb_paths = sorted(rgb_paths)
    
    for i, path in enumerate(tqdm(rgb_paths)):
        rgb_path = os.path.join(data_root_path, path)
        lable_path = rgb_path.replace('rgb', 'label')
        
        inputs = cv2.imread(rgb_path)
        
        # anno = coco_json[i]['annotations']
        # anno = annotations_to_instances(anno, (720, 1280))
        # print(anno)
        # exit()
        
        outputs = predictor(inputs)
        # outputs = model(inputs)
        pred_masks = outputs['instances'].pred_masks.cpu().numpy()
        # gt_masks = anno.gt_masks_bit.tensor.cpu().numpy()
        
        gt_mask = imageio.imread(lable_path)
        pred = np.zeros_like(gt_mask)
        for i, mask in enumerate(pred_masks):
            pred[mask > False] = i + 1

        print("pred & gt size")
        print("sdfsdf")
            
        metrics, assignments = compute_PRF.multilabel_metrics(pred, gt_mask, return_assign=True)
        metrics_all.append(metrics)
        
        #compute IoU for all instances
        num_inst_mat += len(assignments)
        assign_pred, assign_gt = 0, 0
        assign_overlap = 0
        for gt_id, pred_id in assignments:
            gt_mask = gt_mask == gt_id
            pred_mask = pred == pred_id
            
            assign_gt += np.count_nonzero(gt_mask)
            assign_pred += np.count_nonzero(pred_mask)
            
            mask_overlap = np.logical_and(gt_mask, pred_mask)
            assign_overlap += np.count_nonzero(mask_overlap)
        
        if assign_pred + assign_gt - assign_overlap > 0:
            iou = assign_overlap / (assign_pred+assign_gt-assign_overlap)
        else:
            iou = 0
        iou_masks += iou
        
    miou = iou_masks / len(metrics_all)
    
    result = {}
    num = len(metrics_all)
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num
    
    print('\n')
    print(colored("Visible Metrics for OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')
        
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # main(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
