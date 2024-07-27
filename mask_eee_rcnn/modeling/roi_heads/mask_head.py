# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import pycocotools.mask as mask_util

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy
from torchvision.ops import roi_align
from torchvision.ops import deform_conv2d, DeformConv2d

from .mask_encoding import DctMaskEncoding, GT_infomation, patch2masks, masks2patch

from mask_eee_rcnn.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm, DeformConv, DepthwiseConv
from mask_eee_rcnn.utils.events import get_event_storage
from mask_eee_rcnn.utils.registry import Registry
from mask_eee_rcnn.structures.masks import polygons_to_bitmask
from mask_eee_rcnn.structures import Boxes, Instances, pairwise_iou, BitMasks
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from .cbam import CBAM
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, FastRCNNEELayers
from ..box_regression import Box2BoxTransform
from ..poolers import ROIPooler
from ..matcher import Matcher
from .refinemask_loss import BARCrossEntropyLoss, generate_block_target
# from .roi_heads import select_foreground_proposals

import time

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.
The registered object will be called with `obj(cfg, input_shape)`.
"""

def select_foreground_proposals(proposals, bg_label, eee_feat=None, eee_pred=None, box_refine_roi=None):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    if eee_feat == None:
        eee_feat_cat = None
    if eee_pred == None:
        eee_pred_cat = None
    if box_refine_roi == None:
        box_refine_roi_cat = None
    start_idx = 0
    end_idx = 0
    for idx, proposals_per_image in enumerate(proposals):
        gt_classes = proposals_per_image.gt_classes
        proposal_boxes = proposals_per_image.proposal_boxes.tensor
        proposal_width = proposal_boxes[:, 2] - proposal_boxes[:, 0]
        # proposal_height = proposal_boxes[:, 3] - proposal_boxes[:, 1]
        # print(proposal_width)
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label) & (proposal_width > 0)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)

        end_idx += int(gt_classes.shape[0])
        if eee_feat != None:
            # start_idx = int(gt_classes.shape[0] * idx)
            # end_idx = int(gt_classes.shape[0] * (idx + 1))
            eee_feat_per_image = eee_feat[start_idx:end_idx]
            if idx == 0:
                eee_feat_cat = eee_feat_per_image[fg_idxs]
            else:
                eee_feat_cat = torch.cat((eee_feat_cat, eee_feat_per_image[fg_idxs]), dim=0)
        if eee_pred != None:
            # start_idx = int(gt_classes.shape[0] * idx)
            # end_idx = int(gt_classes.shape[0] * (idx + 1))
            eee_pred_per_image = eee_pred[start_idx:end_idx]
            if idx == 0:
                eee_pred_cat = eee_pred_per_image[fg_idxs]
            else:
                eee_pred_cat = torch.cat((eee_pred_cat, eee_pred_per_image[fg_idxs]), dim=0)
        if box_refine_roi != None:
            # start_idx = int(gt_classes.shape[0] * idx)
            # end_idx = int(gt_classes.shape[0] * (idx + 1))
            box_refine_roi_per_image = box_refine_roi[start_idx:end_idx]
            if idx == 0:
                box_refine_roi_cat = box_refine_roi_per_image[fg_idxs]
            else:
                box_refine_roi_cat = torch.cat((box_refine_roi_cat, box_refine_roi_per_image[fg_idxs]), dim=0)
        start_idx = end_idx
    
    return fg_proposals, fg_selection_masks, eee_feat_cat, eee_pred_cat, box_refine_roi_cat

def gt_mask_to_boundary(gt_mask, device):
    gt_mask = gt_mask.to(dtype=torch.float32)
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=device).reshape(1, 1, 3, 3).requires_grad_(False)
    boundary_targets = F.conv2d(gt_mask.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    boundary_targets = boundary_targets.to(dtype=torch.bool)
    boundary_targets = boundary_targets.squeeze(1)
    return boundary_targets

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

def Max(x):
    """
    A wrapper around torch.max in Spatial Attention Module (SAM) to support empty inputs and more features.
    """
    if x.numel() == 0:
        output_shape = [x.shape[0], 1, x.shape[2], x.shape[3]]
        empty = _NewEmptyTensorOp.apply(x, output_shape)
        return empty
    return torch.max(x, dim=1, keepdim=True)[0]

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        weight_init.c2_msra_fill(self.conv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = Max(x)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention), attention


def mask_rcnn_loss(pred_mask_logits, instances, maskiou_on, name="mask_loss", pred_instances=None, images=None, mask_loss_type="ce", combine_dice_ce=False):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    mask_ratios = []
    for idx, instances_per_image in enumerate(instances):
        if len(instances_per_image) == 0:
            continue

        # if not cls_agnostic_mask:
        gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
        gt_classes.append(gt_classes_per_image)

        if maskiou_on:
            if pred_instances is None:
                cropped_mask = instances_per_image.gt_masks.crop(instances_per_image.proposal_boxes.tensor)
                cropped_mask = torch.tensor(
                    [mask_util.area(mask_util.frPyObjects([p for p in obj], box[3]-box[1], box[2]-box[0])).sum().astype(float)
                    for obj, box in zip(cropped_mask.polygons, instances_per_image.proposal_boxes.tensor)]
                    )
            else:
                cropped_mask = instances_per_image.gt_masks.crop(pred_instances[idx].pred_boxes.tensor)
                cropped_mask = torch.tensor(
                    [mask_util.area(mask_util.frPyObjects([p for p in obj], box[3]-box[1], box[2]-box[0])).sum().astype(float)
                    for obj, box in zip(cropped_mask.polygons, pred_instances[idx].pred_boxes.tensor)]
                    )
                
            mask_ratios.append(
                (cropped_mask / instances_per_image.gt_masks.area())
                .to(device=pred_mask_logits.device).clamp(min=0., max=1.)
            )
        
        if pred_instances is None:
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            
            if images is not None:
                import cv2
                print('proposal gt visualizing...')
                for instance_idx in range(instances_per_image.proposal_boxes.tensor.shape[0]):

                    image = images[idx].permute(1,2,0).cpu().numpy()
                    pred_instance_box = instances_per_image.proposal_boxes.tensor[instance_idx].cpu().numpy()
                    
                    gt_mask_per_instance = np.zeros_like(gt_masks_per_image[instance_idx].cpu().numpy())
                    gt_mask_per_instance[gt_masks_per_image[instance_idx].cpu().numpy()] = 1
                    gt_mask_per_instance = np.expand_dims(gt_mask_per_instance, axis=0).astype(np.uint8)
                    gt_mask_per_instance = np.concatenate([gt_mask_per_instance, gt_mask_per_instance, np.zeros_like(gt_mask_per_instance)], axis=0).transpose(1,2,0)
                    gt_mask_per_instance = cv2.resize(gt_mask_per_instance, dsize=(int(pred_instance_box[2])-int(pred_instance_box[0]), int(pred_instance_box[3])-int(pred_instance_box[1])))
                    # gt_mask_per_instance = cv2.rotate(gt_mask_per_instance)
                    
                    cv2.rectangle(image, (int(pred_instance_box[0]), int(pred_instance_box[1])), (int(pred_instance_box[2]), int(pred_instance_box[3])), (0, 255, 0), 2)
                    image[int(pred_instance_box[1]):int(pred_instance_box[3]), int(pred_instance_box[0]):int(pred_instance_box[2])] += gt_mask_per_instance * 200
                    cv2.imwrite('temp_gt_vis/{}_proposal.png'.format(instance_idx), image)

        else:
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                pred_instances[idx].pred_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)

            if images is not None:
                import cv2
                print('refine pred gt visualizing...')
                for instance_idx in range(pred_instances[idx].pred_boxes.tensor.shape[0]):

                    image = images[idx].permute(1,2,0).cpu().numpy()
                    pred_instance_box = pred_instances[idx].pred_boxes.tensor[instance_idx].cpu().numpy()
                    
                    gt_mask_per_instance = np.zeros_like(gt_masks_per_image[instance_idx].cpu().numpy())
                    gt_mask_per_instance[gt_masks_per_image[instance_idx].cpu().numpy()] = 1
                    gt_mask_per_instance = np.expand_dims(gt_mask_per_instance, axis=0).astype(np.uint8)
                    gt_mask_per_instance = np.concatenate([gt_mask_per_instance, gt_mask_per_instance, np.zeros_like(gt_mask_per_instance)], axis=0).transpose(1,2,0)
                    gt_mask_per_instance = cv2.resize(gt_mask_per_instance, dsize=(int(pred_instance_box[2])-int(pred_instance_box[0]), int(pred_instance_box[3])-int(pred_instance_box[1])))
                    # gt_mask_per_instance = cv2.rotate(gt_mask_per_instance)
                    
                    cv2.rectangle(image, (int(pred_instance_box[0]), int(pred_instance_box[1])), (int(pred_instance_box[2]), int(pred_instance_box[3])), (0, 255, 0), 2)
                    image[int(pred_instance_box[1]):int(pred_instance_box[3]), int(pred_instance_box[0]):int(pred_instance_box[2])] += gt_mask_per_instance * 200
                    cv2.imwrite('temp_gt_vis/{}_pred.png'.format(instance_idx), image)

        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)
    # time.sleep(10)

    gt_classes = cat(gt_classes, dim=0)

    if len(gt_masks) == 0:
        if maskiou_on:
            selected_index = torch.arange(pred_mask_logits.shape[0], device=pred_mask_logits.device)
            selected_mask = pred_mask_logits[selected_index, gt_classes]
            mask_num, mask_h, mask_w = selected_mask.shape
            selected_mask = selected_mask.reshape(mask_num, 1, mask_h, mask_w)
            return pred_mask_logits.sum() * 0, selected_mask, gt_classes, None
        
        else:
            return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        # gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar(f"{name}/accuracy", mask_accuracy)
    storage.put_scalar(f"{name}/false_positive", false_positive)
    storage.put_scalar(f"{name}/false_negative", false_negative)

    if combine_dice_ce:
        mask_loss = F.binary_cross_entropy_with_logits(
            pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
        ) * 0.5 
        criterion_dice = DiceLoss(reduction='mean', softmax=False)
        mask_loss += 0.5 * criterion_dice(pred_mask_logits.sigmoid(), gt_masks.to(torch.float32))
    elif mask_loss_type == "dice":
        criterion = DiceLoss(reduction='mean', softmax=False)
        mask_loss = criterion(pred_mask_logits.sigmoid(), gt_masks.to(torch.float32))
    else:
        mask_loss = F.binary_cross_entropy_with_logits(
            pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
        )
    
    if maskiou_on:
        mask_ratios = cat(mask_ratios, dim=0)
        value_eps = 1e-10 * torch.ones(gt_masks.shape[0], device=gt_classes.device)
        mask_ratios = torch.max(mask_ratios, value_eps)

        pred_masks = pred_mask_logits > 0
        
        mask_targets_full_area = gt_masks.sum(dim=[1,2]) / mask_ratios
        # mask_ovr = pred_masks * gt_masks
        mask_ovr_area = (pred_masks * gt_masks).sum(dim=[1,2]).float()
        mask_union_area = pred_masks.sum(dim=[1,2]) + mask_targets_full_area - mask_ovr_area
        value_1 = torch.ones(pred_masks.shape[0], device=gt_classes.device)
        value_0 = torch.zeros(pred_masks.shape[0], device=gt_classes.device)
        mask_union_area = torch.max(mask_union_area, value_1)
        mask_ovr_area = torch.max(mask_ovr_area, value_0)
        maskiou_targets = mask_ovr_area / mask_union_area
        # selected_index = torch.arange(pred_mask_logits.shape[0], device=gt_classes.device)
        # selected_mask = pred_mask_logits[selected_index, gt_classes]
        mask_num, mask_h, mask_w = pred_mask_logits.shape
        selected_mask = pred_mask_logits.reshape(mask_num, 1, mask_h, mask_w)
        selected_mask = selected_mask.sigmoid()
        
        return mask_loss, selected_mask, gt_classes, maskiou_targets.detach()
    else:
        return mask_loss
    

def mask_rcnn_inference(pred_mask_logits, pred_instances, pred_mask_initial_logits=None, pred_eee_logits=None, refined_box_mask_head=None, mask_dct_on=False, patch_dct_on=False, feat_pca=None):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    
    if isinstance(pred_mask_logits, list) and pred_mask_initial_logits is not None:
        pred_mask_logits = pred_mask_logits[-1]
    # if pred_eee_logits is not None:
    #     pred_eee_logits = pred_eee_logits[-1]
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    if pred_mask_initial_logits is not None:
        cls_agnostic_initial_mask = pred_mask_initial_logits.size(1) == 1

    if refined_box_mask_head is not None:
        for idx, instances in enumerate(pred_instances):
            # i.pred_boxes = refined_box_mask_head[i].pred_boxes
            instances.pred_boxes_initial = instances.pred_boxes
            instances.scores_initial = instances.scores
            instances.pred_classes_initial = instances.pred_classes
            instances.pred_boxes = refined_box_mask_head[idx].pred_boxes
            if refined_box_mask_head[idx].has("scores"):
                instances.scores = refined_box_mask_head[idx].scores
                instances.pred_classes = refined_box_mask_head[idx].pred_classes

    if cls_agnostic_mask:
        if mask_dct_on or patch_dct_on:
            mask_probs_pred = pred_mask_logits
        else:
            mask_probs_pred = pred_mask_logits.sigmoid()
        
        if pred_mask_initial_logits is not None:
            mask_probs_pred_initial = pred_mask_initial_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        try:
            mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        except IndexError:
            mask_probs_pred = pred_mask_logits[indices,:][:, None].sigmoid()
        if pred_mask_initial_logits is not None:
            if cls_agnostic_initial_mask:
                mask_probs_pred_initial = pred_mask_initial_logits.sigmoid()
            else:
                mask_probs_pred_initial = pred_mask_initial_logits[indices, class_pred][:, None].sigmoid()
    
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)
    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    if pred_mask_initial_logits is not None:
        mask_probs_pred_initial = mask_probs_pred_initial.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)
    
    if pred_mask_initial_logits is not None:
        for prob_initial, instances in zip(mask_probs_pred_initial, pred_instances):
            instances.pred_masks_initial = prob_initial  # (1, Hmask, Wmask)

    if pred_eee_logits == None or pred_eee_logits.nelement() == 0:
        pred_eee_logits = None
    if pred_eee_logits is not None:
        num_boxes_per_image = [len(i) for i in pred_instances]
        # print(torch.unique(pred_eee_logits))
        # pred_eee_logits = pred_eee_logits.argmax(dim=1)
        pred_eee_logits = pred_eee_logits.softmax(dim=1)
        pred_eee_logits = pred_eee_logits[:,1]
        pred_eee_logits = pred_eee_logits.split(num_boxes_per_image, dim=0)
        for eee, instances in zip(pred_eee_logits, pred_instances):
            instances.pred_errors = eee
    
    if feat_pca is not None:
        num_boxes_per_image = [len(i) for i in pred_instances]
        if feat_pca[0] is not None:
            coarse_feat_pca = feat_pca[0]
            coarse_feat_pca = coarse_feat_pca.split(num_boxes_per_image, dim=0)
            for feat, instances in zip(coarse_feat_pca, pred_instances):
                instances.coarse_feat_pca = feat
        
        if feat_pca[1] is not None:
            eee_feat_pca = feat_pca[1]
            eee_feat_pca = eee_feat_pca.split(num_boxes_per_image, dim=0)
            for feat, instances in zip(eee_feat_pca, pred_instances):
                instances.eee_feat_pca = feat

        if feat_pca[2] is not None:
            refine_feat_pca = feat_pca[2]
            refine_feat_pca = refine_feat_pca.split(num_boxes_per_image, dim=0)
            for feat, instances in zip(refine_feat_pca, pred_instances):
                instances.refine_feat_pca = feat
            

def mask_rcnn_eee_loss(pred_mask_logits, pred_eee_logits, instances, loss_type='dice', error_type='e3', combine_dice_ce=False, boundary_error_on=False, pred_instances=None):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    if error_type == 'e2':
        cls_agnostic_error = pred_eee_logits.size(1) == 2
    elif error_type == 'e3':
        cls_agnostic_error = pred_eee_logits.size(1) == 4
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for idx, instances_per_image in enumerate(instances):
        if len(instances_per_image) == 0:
            continue

        # if not cls_agnostic_mask:
        gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
        gt_classes.append(gt_classes_per_image)

        if pred_instances is None:
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        else:
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                pred_instances[idx].pred_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    gt_classes = cat(gt_classes, dim=0)
    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        # gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
    pred_mask_bool = pred_mask_logits > 0
    
    if len(gt_masks) == 0:
        print('no gt mask')
        gt_masks_bool = torch.zeros_like(pred_mask_bool)
        true_positive_mask =  gt_masks_bool & pred_mask_bool
        true_negative_mask = gt_masks_bool & ~pred_mask_bool
        false_positive_mask = ~gt_masks_bool & pred_mask_bool
        false_negative_mask = ~gt_masks_bool & ~pred_mask_bool
        selected_mask = pred_mask_logits.reshape(-1, 1, mask_side_len, mask_side_len)
        return pred_mask_logits.sum() * 0, pred_eee_logits.sum() * 0, selected_mask, true_positive_mask, true_negative_mask, false_positive_mask, false_negative_mask
    
    gt_masks = cat(gt_masks, dim=0)
    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    if cls_agnostic_error:
        pred_eee_logits = pred_eee_logits
    else:
        # indices = torch.arange(total_num_masks)
        # gt_classes = cat(gt_classes, dim=0)
        for gt_idx, gt_class in enumerate(gt_classes):
            pred_eee_logits_class = pred_eee_logits[gt_idx, 2*gt_class:(2*gt_class+2)].unsqueeze(0)
            if gt_idx == 0:
                pred_eee_logits_filter = pred_eee_logits_class
            else:
                pred_eee_logits_filter = torch.cat([pred_eee_logits_filter, pred_eee_logits_class], dim=0)
        pred_eee_logits = pred_eee_logits_filter

    # import cv2
    # import numpy as np
    # n_masks = pred_mask_bool.shape[0]
    # for i in range(n_masks):
    #     vis = pred_mask_bool[i].cpu().numpy().astype(np.uint8)
    #     vis = np.stack([vis, vis, vis], axis=2)
    #     vis = vis * 255
    #     cv2.imwrite('{}_pred.png'.format(i), vis)


    # if 'boundary' in loss_type:
    if boundary_error_on:
        device = gt_masks_bool.device
        gt_masks_bool = gt_mask_to_boundary(gt_masks_bool, device=device)
        pred_mask_bool = gt_mask_to_boundary(pred_mask_bool, device=device)

    true_positive_mask = gt_masks_bool & pred_mask_bool
    true_negative_mask = ~gt_masks_bool & ~pred_mask_bool
    false_positive_mask = ~gt_masks_bool & pred_mask_bool
    false_negative_mask = gt_masks_bool & ~pred_mask_bool

    # import numpy as np
    # import cv2
    # n_masks = false_negative_mask.shape[0]
    # for idx in range(n_masks):
    #     tp = true_positive_mask[idx].detach().cpu().numpy().astype(np.uint8) * 255
    #     fp = false_positive_mask[idx].detach().cpu().numpy().astype(np.uint8) * 255
    #     fn = false_negative_mask[idx].detach().cpu().numpy().astype(np.uint8) * 255

    #     vis = np.zeros([tp.shape[0], tp.shape[1], 3])
    #     vis[:, :, 0] = fn
    #     vis[:, :, 1] = tp
    #     vis[:, :, 2] = fp
    #     vis = vis.astype(np.uint8)
    #     cv2.imwrite('vis/{}_boundary_gt.png'.format(idx), vis)
    if error_type == 'e3':
        gt_mask = torch.cat([
                    true_positive_mask.unsqueeze(1),
                    true_negative_mask.unsqueeze(1),
                    false_positive_mask.unsqueeze(1),
                    false_negative_mask.unsqueeze(1),
                ], dim=1).to(dtype=torch.float) # [N, 4, H, W]
    elif error_type == 'e2':
        true_mask = true_positive_mask | true_negative_mask
        false_mask = false_positive_mask | false_negative_mask
        gt_mask = torch.cat([
                    true_mask.unsqueeze(1),
                    false_mask.unsqueeze(1),
                ], dim=1).to(dtype=torch.float)
    elif error_type == 'e33':
        true_mask = true_positive_mask | true_negative_mask
        gt_mask = torch.cat([
                    true_mask.unsqueeze(1),
                    false_positive_mask.unsqueeze(1),
                    false_negative_mask.unsqueeze(1),
                ], dim=1).to(dtype=torch.float)
    elif error_type == 'e32':
        gt_mask = torch.cat([
                    false_positive_mask.unsqueeze(1),
                    false_negative_mask.unsqueeze(1),
                ], dim=1).to(dtype=torch.float)

    if combine_dice_ce:
        criterion_ce = nn.CrossEntropyLoss(reduction='mean')
        criterion_dice = DiceLoss(reduction='mean', softmax=True)
    elif loss_type == 'dice':
        criterion = DiceLoss(reduction='mean', softmax=True)
    elif loss_type == 'dicece':
        criterion = DiceCELoss(reduction='mean', softmax=True)
    elif loss_type == 'dicefocal':
        criterion = DiceFocalLoss(reduction='mean', softmax=True)
    elif loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    
    if combine_dice_ce:
        mask_eee_loss = 0.5 * criterion_dice(pred_eee_logits, gt_mask) + 0.5 * criterion_ce(pred_eee_logits, gt_mask.long())
    elif loss_type == 'ce':
        mask_eee_loss = criterion(pred_eee_logits, gt_mask.long())
    else:
        mask_eee_loss = criterion(pred_eee_logits, gt_mask)
    return mask_eee_loss


def mask_rcnn_dct_loss(pred_mask_logits, instances, dct_encoding, pred_instances=None, vis_period=0, dct_loss_type='l1', mask_size=128, mask_loss_para=1.0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): [B, D]. D is dct-dim. [B, D]. DCT_Vector.
        
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    
    gt_masks = []
    for idx, instances_per_image in enumerate(instances):
        if len(instances_per_image) == 0:
            continue
        
        if pred_instances is None:
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_size)
            gt_masks_vector = dct_encoding.encode(gt_masks_per_image)  # [N, dct_v_dim]
        else:
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                pred_instances[idx].pred_boxes.tensor, mask_size)
            gt_masks_vector = dct_encoding.encode(gt_masks_per_image)
        gt_masks.append(gt_masks_vector)


    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)
    
    gt_masks = gt_masks.to(dtype=torch.float32)
    if dct_loss_type == "l1":
        num_instance = gt_masks.size()[0]
        mask_loss = F.l1_loss(pred_mask_logits, gt_masks, reduction="none")
        mask_loss = mask_loss_para * mask_loss / num_instance
        mask_loss = torch.sum(mask_loss)
        
    elif dct_loss_type == "sl1":
        num_instance = gt_masks.size()[0]
        mask_loss = F.smooth_l1_loss(pred_mask_logits, gt_masks, reduction="none")
        mask_loss = mask_loss_para * mask_loss / num_instance
        mask_loss = torch.sum(mask_loss)
    elif dct_loss_type == "l2":
        num_instance = gt_masks.size()[0]
        mask_loss = F.mse_loss(pred_mask_logits, gt_masks, reduction="none")
        mask_loss = mask_loss_para * mask_loss / num_instance
        mask_loss = torch.sum(mask_loss)
    else:
        raise ValueError("Loss Type Only Support : l1, l2; yours: {}".format(dct_loss_type))

    return mask_loss


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


def boundary_loss_func(boundary_logits, gtmasks):
    """
    Args:
        boundary_logits (Tensor): A tensor of shape (B, H, W) or (B, H, W)
        gtmasks (Tensor): A tensor of shape (B, H, W) or (B, H, W)
    """
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=boundary_logits.device).reshape(1, 1, 3, 3).requires_grad_(False)
    boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0

    if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
        boundary_targets = F.interpolate(
            boundary_targets, boundary_logits.shape[2:], mode='nearest')

    bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
    dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boundary_targets)
    return bce_loss + dice_loss


def boundary_preserving_mask_loss(
        pred_mask_logits,
        pred_boundary_logits,
        instances,
        pred_instances=None,
        vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for idx, instances_per_image in enumerate(instances):
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if pred_instances is None:
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)
        else:
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                pred_instances[idx].pred_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0, pred_boundary_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        pred_boundary_logits = pred_boundary_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
        pred_boundary_logits = pred_boundary_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    boundary_loss = boundary_loss_func(pred_boundary_logits, gt_masks)
    return mask_loss, boundary_loss


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class MultiBranchFusion(nn.Module):

    def __init__(self, feat_dim, dilations=[1, 3, 5]):
        super(MultiBranchFusion, self).__init__()

        for idx, dilation in enumerate(dilations):
            self.add_module(f'dilation_conv_{idx + 1}', Conv2d(
                feat_dim, feat_dim, kernel_size=3, padding=dilation, dilation=dilation))

        # self.merge_conv = Conv2d(feat_dim, feat_dim, kernel_size=1, act_cfg=None)
        self.merge_conv = Conv2d(feat_dim, feat_dim, kernel_size=1)

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3)
        return out_feat


class MultiBranchFusionAvg(MultiBranchFusion):

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        feat_4 = F.avg_pool2d(x, x.shape[-1])
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3 + feat_4)
        return out_feat


class SimpleSFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 instance_in_channel=256,
                 instance_out_channel=256,
                 fusion_type='MultiBranchFusion',
                 dilations=[1, 3, 5],
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 upsample_cfg=dict(type='bilinear', scale_factor=2)):
        super(SimpleSFMStage, self).__init__()

        self.semantic_out_stride = semantic_out_stride
        # self.mask_use_sigmoid = mask_use_sigmoid
        self.num_classes = num_classes

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        self.semantic_roi_extractor = ROIPooler(
                    # output_size=pooler_resolution * 4, # 7x7 -> 28x28
                    output_size=out_size,
                    scales=[1/semantic_out_stride, ],
                    sampling_ratio=0,
                    pooler_type="ROIAlignV2",
                )

        fuse_in_channel = instance_in_channel + semantic_out_channel + 1
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1),
            globals()[fusion_type](instance_in_channel, dilations=dilations)])

        self.fuse_transform_out = nn.Conv2d(instance_in_channel, instance_out_channel - 1, 1)
        # self.upsample = build_upsample_layer(upsample_cfg.copy())
        # self.upsample = nn.Upsample(upsample_cfg.copy())
        self.upsample = nn.Upsample(mode="bilinear", scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, instance_logits, semantic_feat, rois, upsample=True):

        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))
        ins_semantic_feats = self.semantic_roi_extractor([semantic_feat,], rois)

        # fuse instance feats & instance masks & semantic feats & semantic masks
        concat_tensors = [instance_feats, ins_semantic_feats, instance_logits.sigmoid()]
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        fused_feats = torch.cat([fused_feats, instance_logits.sigmoid()], dim=1)
        fused_feats = self.upsample(fused_feats) if upsample else fused_feats
        # fused_feats = self.relu(self.upsample(fused_feats))

        return fused_feats


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec, input_shape_pooler):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        num_fusion_conv   = cfg.MODEL.ROI_MASK_HEAD.NUM_FUSION_CONV
        num_initial_conv  = cfg.MODEL.ROI_MASK_HEAD.NUM_INITIAL_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.mask_initial_on = cfg.MODEL.ROI_MASK_HEAD.MASK_INITIAL_ON
        self.mask_eee_on = cfg.MODEL.ROI_MASK_HEAD.MASK_EEE_ON
        self.mask_refine_on = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_ON
        self.mask_refine_num = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_NUM
        self.mask_refine_size = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_SIZE
        self.mask_refine_add_size = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_ADD_SIZE
        self.mask_refine_depthwise_conv = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_DEPTHWISE_CONV
        self.fusion_targets = cfg.MODEL.ROI_MASK_HEAD.FUSION_TARGETS
        self.error_type = cfg.MODEL.ROI_MASK_HEAD.MASK_EEE_ERROR_TYPE
        self.deform_num_groups = cfg.MODEL.ROI_MASK_HEAD.DEFORM_NUM_GROUPS
        self.self_attention_type = cfg.MODEL.ROI_MASK_HEAD.SELF_ATTENTION_TYPE
        self.stop_gradient = cfg.MODEL.ROI_MASK_HEAD.STOP_GRADIENT
        self.dense_fusion_on = cfg.MODEL.ROI_MASK_HEAD.DENSE_FUSION_ON
        self.eee_fusion_type = cfg.MODEL.ROI_MASK_HEAD.EEE_FUSION_TYPE
        self.spatial_attention_on = cfg.MODEL.ROI_MASK_HEAD.SPATIAL_ATTENTION_ON
        self.box_refine_on_mask_head = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_HEAD
        self.box_refine_on_mask_head_error_fusion = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_HEAD_ERROR_FUSION
        self.box_refine_feat_fusion = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_FEAT_FUSION
        self.box_refine_on_mask_new_pooler = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_NEW_POOLER
        self.box_refine_align_past_feature = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ALIGN_PAST_FEATURE
        self.box_refine_on_mask_head_box_class_agnostic = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_HEAD_BOX_CLASS_AGNOSTIC
        self.mask_feat_size = cfg.MODEL.ROI_MASK_HEAD.MASK_FEAT_SIZE
        self.box_refine_on_mask_head_class_refine = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_HEAD_CLASS_REFINE
        self.box_refine_on_mask_head_class_refine_after_boxfix = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_HEAD_CLASS_REFINE_AFTER_BOXFIX
        self.box_refine_on_mask_head_class_refine_with_maskfix = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_HEAD_CLASS_REFINE_WITH_MASKFIX
        self.box_refine_on_mask_head_class_refine_after_maskfix = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_HEAD_CLASS_REFINE_AFTER_MASKFIX
        self.cls_refine_loss_weight = cfg.MODEL.ROI_MASK_HEAD.CLS_REFINE_LOSS_WEIGHT
        self.box_to_mask_feature_fusion = cfg.MODEL.ROI_BOX_HEAD.BOX_TO_MASK_FEATURE_FUSION
        self.box_to_boxfix_feature_fusion = cfg.MODEL.ROI_BOX_HEAD.BOX_TO_BOXFIX_FEATURE_FUSION
        self.box_to_boxfix_fc_feature_fusion = cfg.MODEL.ROI_BOX_HEAD.BOX_TO_BOXFIX_FC_FEATURE_FUSION
        self.box_to_boxfix_detach = cfg.MODEL.ROI_BOX_HEAD.BOX_TO_BOXFIX_DETACH
        self.cls_logits_to_mask_feature_fusion = cfg.MODEL.ROI_BOX_HEAD.CLS_LOGITS_TO_MASK_FEATURE_FUSION
        self.cls_logits_to_box_fix_directly = cfg.MODEL.ROI_BOX_HEAD.CLS_LOGITS_TO_BOX_FIX_DIRECTLY
        self.cls_logits_to_mb_detach = cfg.MODEL.ROI_BOX_HEAD.CLS_LOGITS_TO_MB_DETACH
        self.cls_logits_to_mask_normalize_fusion = cfg.MODEL.ROI_BOX_HEAD.CLS_LOGITS_TO_MASK_NORMALIZE_FUSION
        self.num_classes_for_normalize  = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.mask_refine_on_proposal_box = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_ON_PROPOSAL_BOX
        self.box_refine_on_mask_random_roi_training = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_RANDOM_ROI_TRAINING
        self.mask_refine_on_gt_roi_training = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_ON_GT_ROI_TRAINING
        self.mask_refine_on_gt_roi_randomly = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_ON_GT_ROI_RANDOMLY
        self.error_estimation_class_agnostic = cfg.MODEL.ROI_MASK_HEAD.ERROR_ESTIMATION_CLASS_AGNOSTIC
        self.error_estimation_fusion_class_agnostic = cfg.MODEL.ROI_MASK_HEAD.ERROR_ESTIMATION_FUSION_CLASS_AGNOSTIC
        self.fusion_pred_class_only = cfg.MODEL.ROI_MASK_HEAD.FUSION_PRED_CLASS_ONLY
        self.mask_dct_on = cfg.MODEL.ROI_MASK_HEAD.MASK_DCT_ON
        self.mask_dct_after_maskfix_on = cfg.MODEL.ROI_MASK_HEAD.MASK_DCT_AFTER_MASKFIX_ON
        self.deconv_after_maskfix = cfg.MODEL.ROI_MASK_HEAD.DECONV_AFTER_MASKFIX
        self.deconv_before_maskfix = cfg.MODEL.ROI_MASK_HEAD.DECONV_BEFORE_MASKFIX
        self.semantic_fusion_module_on = cfg.MODEL.ROI_MASK_HEAD.SEMANTIC_FUSION_MODULE_ON
        self.maskfix_semantic_fusion_module_on = cfg.MODEL.ROI_MASK_HEAD.MASKFIX_SEMANTIC_FUSION_MODULE_ON
        self.cls_agnostic_initial_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_INITIAL_MASK
        self.boundary_preserving_on = cfg.MODEL.ROI_MASK_HEAD.BOUNDARY_PRESERVING_ON
        self.refinemask_on = cfg.MODEL.ROI_MASK_HEAD.REFINEMASK_ON
        self.fusion_boxfix_feat_to_maskfix = cfg.MODEL.ROI_MASK_HEAD.FUSION_BOXFIX_FEAT_TO_MASKFIX
        self.fusion_boxfix_feat_to_maskfix_layer_idx = cfg.MODEL.ROI_MASK_HEAD.FUSION_BOXFIX_FEAT_TO_MASKFIX_LAYER_IDX
        self.boundary_error_on = cfg.MODEL.ROI_MASK_HEAD.BOUNDARY_ERROR_ON
        self.maskfix_fusion_with_deform_conv = cfg.MODEL.ROI_MASK_HEAD.MASKFIX_FUSION_WITH_DEFORM_CONV
        self.mask_scoring_at_error_estimation = cfg.MODEL.ROI_MASK_HEAD.MASK_SCORING_AT_ERROR_ESTIMATION
        self.mask_scoring_at_error_estimation_fc = cfg.MODEL.ROI_MASK_HEAD.MASK_SCORING_AT_ERROR_ESTIMATION_FC
        self.mask_scoring_at_error_estimation_fc_pred_only = cfg.MODEL.ROI_MASK_HEAD.MASK_SCORING_AT_ERROR_ESTIMATION_FC_PRED_ONLY
        self.patchdct_on = cfg.MODEL.ROI_MASK_HEAD.PATCHDCT_ON
        self.fusion_with_depthwise_conv = cfg.MODEL.ROI_MASK_HEAD.FUSION_WITH_DEPTHWISE_CONV
        self.error_fusion_maskfix = cfg.MODEL.ROI_MASK_HEAD.ERROR_FUSION_MASKFIX
        self.transfiner_on = cfg.MODEL.ROI_MASK_HEAD.TRANSFINER_ON
        self.use_cascade_iou_thresh = cfg.MODEL.ROI_MASK_HEAD.USE_CASCADE_IOU_THRESH
        
        see_conv_dims = cfg.MODEL.ROI_MASK_HEAD.SEE_CONV_CHANNEL
        boxfix_conv_dims = cfg.MODEL.ROI_MASK_HEAD.BOXFIX_CONV_CHANNEL
        maskfix_conv_dims = cfg.MODEL.ROI_MASK_HEAD.MASKFIX_CONV_CHANNEL
        box_conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        b2b_feature_fusion_num = cfg.MODEL.ROI_MASK_HEAD.B2B_FEATURE_FUSION_NUM

        self.sem_seg_to_dualfix = cfg.MODEL.ROI_MASK_HEAD.SEM_SEG_TO_DUALFIX
        self.sem_seg_on = cfg.MODEL.SEM_SEG_HEAD.SEM_SEG_ON
        sem_seg_conv_dims = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM

        # fmt: on

        if num_initial_conv == 0:
            num_initial_conv = num_conv

        if self.mask_eee_on or self.mask_refine_on:
            assert num_fusion_conv >= 0 , "num_fusion_conv must be > 0"
        
        if self.mask_initial_on:
            # initial mask prediction layers
            self.conv_norm_relus_list = []
            self.predictor_list = []
            for mask_refine_idx in range(self.mask_refine_num):
                conv_norm_relus = []
                for k in range(num_initial_conv):
                    initial_channels = input_channels
                    if mask_refine_idx > 0:
                        if self.mask_eee_on:
                            initial_channels += see_conv_dims
                            initial_channels += 2
                        else:
                            initial_channels += conv_dims
                            initial_channels += num_classes
                    # if self.sem_seg_on:
                    #     initial_channels += num_classes
                    #     initial_channels += sem_seg_conv_dims
                    conv = Conv2d(
                        initial_channels if k == 0 else conv_dims,
                        conv_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not self.norm,
                        norm=get_norm(self.norm, conv_dims),
                        activation=F.relu,
                    )
                    if mask_refine_idx == 0:
                        self.add_module("mask_fcn{}".format(k + 1), conv)
                    else:
                        self.add_module("mask_fcn{}_{}".format(mask_refine_idx, k + 1), conv)
                    # self.add_module("mask_fcn{}".format(k + 1), conv)
                    conv_norm_relus.append(conv)
                self.conv_norm_relus_list.append(conv_norm_relus)

                for layer in conv_norm_relus:
                    weight_init.c2_msra_fill(layer)

                if not self.mask_dct_on:
                    if self.mask_refine_size != 14:
                        self.deconv = ConvTranspose2d(
                            conv_dims if num_initial_conv > 0 else input_channels,
                            conv_dims,
                            kernel_size=2,
                            stride=2,
                            padding=0,
                        )
                        weight_init.c2_msra_fill(self.deconv)

                    num_initial_mask_classes = 1 if cls_agnostic_mask or self.cls_agnostic_initial_mask else num_classes
                    
                    predictor = Conv2d(conv_dims, num_initial_mask_classes, kernel_size=1, stride=1, padding=0)
                    if mask_refine_idx == 0:
                        self.add_module("predictor", predictor)
                    else:
                        self.add_module("predictor_{}".format(mask_refine_idx), predictor)
                    # if self.mask_refine_num == 1:
                    #     self.predictor = predictor
                    # else:
                    self.predictor_list.append(predictor)

                    # use normal distribution initialization for mask prediction layer
                    nn.init.normal_(predictor.weight, std=0.001)
                    if predictor.bias is not None:
                        nn.init.constant_(predictor.bias, 0)


        if self.mask_eee_on:
            
            if self.error_type == 'e3':
                error_dim = 4
            elif self.error_type == 'e2':
                error_dim = 2
            elif self.error_type == 'e33':
                error_dim = 3
            elif self.error_type == 'e32':
                error_dim = 2

            self.conv_norm_relus_i2e_fusion_list = []
            self.conv_norm_relus_eee_list = []
            self.predictor_eee_list = []
            for mask_refine_idx in range(self.mask_refine_num):
                conv_norm_relus_i2e_fusion = []
                initial_channel = input_channels
                if "feat" in self.fusion_targets and self.eee_fusion_type == 'cat':
                    initial_channel += conv_dims
                if "pred" in self.fusion_targets:
                    if cls_agnostic_mask or self.cls_agnostic_initial_mask:
                        initial_channel += 1
                    elif self.fusion_pred_class_only:
                        initial_channel += 1
                    elif self.mask_dct_on:
                        initial_channel += 1
                    else:
                        initial_channel += num_classes
                
                for k in range(num_fusion_conv):
                    # 1x1 conv to reduce the channel dimension
                    # x_l (input_channels) + mask_feat (conv_dims) + mask_pred (1)
                    # then 3x3 conv to fuse the features
                    # !TODO: support argmax prediction
                    if self.fusion_with_depthwise_conv:
                        conv = DepthwiseConv(
                            initial_channel if k == 0 else see_conv_dims,
                            see_conv_dims,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=not self.norm,
                            norm=get_norm(self.norm, see_conv_dims),
                            activation=F.relu,
                        )
                    else:
                        conv = Conv2d(
                            initial_channel if k == 0 else conv_dims,
                            conv_dims,
                            kernel_size=1 if k == 0 else 3,
                            stride=1,
                            padding=0 if k == 0 else 1,
                            bias=not self.norm,
                            norm=get_norm(self.norm, conv_dims),
                            activation=F.relu,
                        )
                    if mask_refine_idx == 0:
                        self.add_module("mask_i2e_fusion{}".format(k + 1), conv)
                    else:
                        self.add_module("mask_i2e_fusion{}_{}".format(mask_refine_idx, k + 1), conv)
                    # self.add_module("mask_i2e_fusion{}".format(k + 1), conv)
                    conv_norm_relus_i2e_fusion.append(conv)
                self.conv_norm_relus_i2e_fusion_list.append(conv_norm_relus_i2e_fusion)

                conv_norm_relus_eee = []
                for k in range(num_conv):
                    conv = Conv2d(
                        initial_channel if k == 0 and (num_fusion_conv == 0) else see_conv_dims,
                        see_conv_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not self.norm,
                        norm=get_norm(self.norm, conv_dims),
                        activation=F.relu,
                    )
                    if mask_refine_idx == 0:
                        self.add_module("mask_fcn_eee{}".format(k + 1), conv)
                    else:
                        self.add_module("mask_fcn_eee{}_{}".format(mask_refine_idx, k + 1), conv)
                    # self.add_module("mask_fcn_eee{}".format(k + 1), conv)
                    conv_norm_relus_eee.append(conv)
                self.conv_norm_relus_eee_list.append(conv_norm_relus_eee)
                if self.error_estimation_class_agnostic:
                    predictor_eee = Conv2d(see_conv_dims, error_dim, kernel_size=1, stride=1, padding=0)
                else:
                    predictor_eee = Conv2d(see_conv_dims, error_dim * num_classes, kernel_size=1, stride=1, padding=0)
                
                if self.mask_refine_num == 1:
                    self.predictor_eee = predictor_eee
                    self.add_module("mask_fcn_eee_pred", self.predictor_eee)
                else:
                    self.predictor_eee_list.append(predictor_eee)
                    if mask_refine_idx == 0:
                        self.add_module("mask_fcn_eee_pred", predictor_eee)
                    else:
                        self.add_module("mask_fcn_eee_pred_{}".format(mask_refine_idx), predictor_eee)

                for layer in conv_norm_relus_i2e_fusion + conv_norm_relus_eee:
                    if type(layer) == DepthwiseConv:
                        weight_init.c2_msra_fill(layer.depthwise)
                        weight_init.c2_msra_fill(layer.pointwise)
                    else:
                        weight_init.c2_msra_fill(layer)
                # use normal distribution initialization for mask prediction layer
                nn.init.normal_(predictor_eee.weight, std=0.001)
                if predictor_eee.bias is not None:
                    nn.init.constant_(predictor_eee.bias, 0)


        if self.mask_refine_on:
            self.conv_norm_relus_e2r_fusion_list = []
            self.conv_norm_relus_refine_list = []
            mask_refine_idx = 0
            conv_norm_relus_e2r_fusion = []
            initial_channel = input_channels
            
            if self.error_fusion_maskfix:
                if "feat" in self.fusion_targets and self.eee_fusion_type == 'cat':
                    if self.mask_eee_on:
                        initial_channel += see_conv_dims
                    else:
                        initial_channel += conv_dims
                if "pred" in self.fusion_targets:
                    if self.mask_eee_on:
                        if self.error_estimation_class_agnostic:
                            initial_channel += error_dim
                        else:
                            initial_channel += error_dim * num_classes
                    else:
                        initial_channel += num_classes
            if self.dense_fusion_on:
                if "feat" in self.fusion_targets:
                    initial_channel += conv_dims
                if "pred" in self.fusion_targets:
                    initial_channel += num_classes

            for k in range(num_fusion_conv):
                # 1x1 conv to reduce the channel dimension
                # x_l (input_channels) + mask_feat (conv_dims) + eee_pred (4)
                # then 3x3 conv to fuse the features
                if self.fusion_with_depthwise_conv:
                    conv = DepthwiseConv(
                        initial_channel if k == 0 else maskfix_conv_dims,
                        maskfix_conv_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not self.norm,
                        norm=get_norm(self.norm, maskfix_conv_dims),
                        activation=F.relu,
                    )
                else:
                    conv = Conv2d(
                        initial_channel if k == 0 else maskfix_conv_dims,
                        maskfix_conv_dims,
                        kernel_size=1 if k == 0 else 3,
                        stride=1,
                        padding=0 if k == 0 else 1,
                        bias=not self.norm,
                        norm=get_norm(self.norm, maskfix_conv_dims),
                        activation=F.relu,
                    )
                if mask_refine_idx == 0:
                    self.add_module("mask_e2r_fusion{}".format(k + 1), conv)
                else:
                    self.add_module("mask_e2r_fusion{}_{}".format(mask_refine_idx, k + 1), conv)
                # self.add_module("mask_e2r_fusion{}".format(k + 1), conv)
                conv_norm_relus_e2r_fusion.append(conv)
            self.conv_norm_relus_e2r_fusion_list.append(conv_norm_relus_e2r_fusion)
            
            if self.refinemask_on:
                maskfix_num_conv = 2
            else:
                maskfix_num_conv = 4

            self.conv_norm_relus_refine = []
            for k in range(maskfix_num_conv):
                if self.maskfix_fusion_with_deform_conv and k == 0:
                    self.maskfix_deformable_offset = Conv2d(
                        initial_channel if k == 0 and (num_fusion_conv == 0) else maskfix_conv_dims,
                        # maskfix_conv_dims * 2,
                        3 * 3 * 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not self.norm,
                        norm=get_norm(self.norm, maskfix_conv_dims),
                        activation=F.relu,
                    )
                    weight_init.c2_msra_fill(self.maskfix_deformable_offset)
                    conv = DeformConv2d(
                        initial_channel if k == 0 and (num_fusion_conv == 0) else maskfix_conv_dims,
                        maskfix_conv_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not self.norm,
                        # norm=get_norm(self.norm, maskfix_conv_dims),
                        # activation=F.relu,
                    )
                else:
                    conv = Conv2d(
                        initial_channel if k == 0 and (num_fusion_conv == 0) else maskfix_conv_dims,
                        maskfix_conv_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not self.norm,
                        norm=get_norm(self.norm, maskfix_conv_dims),
                        activation=F.relu,
                    )
                # if mask_refine_idx == 0:
                #     self.add_module("mask_fcn_refine{}".format(k + 1), conv)
                # else:
                #     self.add_module("mask_fcn_refine{}_{}".format(mask_refine_idx, k + 1), conv)
                self.add_module("mask_fcn_refine{}".format(k + 1), conv)
                self.conv_norm_relus_refine.append(conv)
            # self.conv_norm_relus_refine_list.append(conv_norm_relus_refine)
            
            for layer in conv_norm_relus_e2r_fusion + self.conv_norm_relus_refine:
                if type(layer) == DepthwiseConv:
                    weight_init.c2_msra_fill(layer.depthwise)
                    weight_init.c2_msra_fill(layer.pointwise)
            else:
                weight_init.c2_msra_fill(layer)
        
            if not self.mask_dct_after_maskfix_on and not self.refinemask_on and not self.patchdct_on:
                num_mask_classes = 1 if cls_agnostic_mask else num_classes
                self.predictor_refine = Conv2d(maskfix_conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
                self.add_module("mask_fcn_refine", self.predictor_refine)
                # self.add_module("mask_fcn_refine", predictor_refine)
                # use normal distribution initialization for mask prediction layer
                nn.init.normal_(self.predictor_refine.weight, std=0.001)
                if self.predictor_refine.bias is not None:
                    nn.init.constant_(self.predictor_refine.bias, 0)

        if self.deconv_after_maskfix:
            self.deconv_after_maskfix_layer = ConvTranspose2d(
                maskfix_conv_dims,
                maskfix_conv_dims,
                kernel_size=2,
                stride=2,
                padding=0,
            )
            weight_init.c2_msra_fill(self.deconv_after_maskfix_layer)

        if self.deconv_before_maskfix:
            self.deconv_before_maskfix_layer = ConvTranspose2d(
                conv_dims,
                conv_dims,
                kernel_size=2,
                stride=2,
                padding=0,
            )
            weight_init.c2_msra_fill(self.deconv_before_maskfix_layer)

            self.deconv_before_maskfix_layer_pred = ConvTranspose2d(
                error_dim,
                error_dim,
                kernel_size=2,
                stride=2,
                padding=0,
            )
            weight_init.c2_msra_fill(self.deconv_before_maskfix_layer_pred)

            if self.dense_fusion_on:
                self.deconv_before_maskfix_layer_dense = ConvTranspose2d(
                    conv_dims,
                    conv_dims,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
                weight_init.c2_msra_fill(self.deconv_before_maskfix_layer_dense)

                self.deconv_before_maskfix_layer_dense_pred = ConvTranspose2d(
                    1 if self.fusion_pred_class_only or self.cls_agnostic_initial_mask else num_classes,
                    1 if self.fusion_pred_class_only or self.cls_agnostic_initial_mask else num_classes,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
                weight_init.c2_msra_fill(self.deconv_before_maskfix_layer_dense_pred)
                    
        if self.box_refine_on_mask_head:
                
            box_conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
            # num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
            boxfix_num_fc = cfg.MODEL.ROI_MASK_HEAD.BOXFIX_NUM_FC
            fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
            
            box_num_fusion_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_FUSION_CONV_FC

            self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
            self.feature_strides          = {k: v.stride for k, v in input_shape_pooler.items()}
            self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
            self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
            self.box_refine_loss_weight   = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_LOSS_WEIGHT
            if self.box_refine_loss_weight == None:
                self.box_refine_loss_weight   = cfg.MODEL.ROI_BOX_HEAD.BOX_REFINE_LOSS_WEIGHT

            # pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            # pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
            # sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
            # pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
            pooler_resolution   = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
            pooler_scales       = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
            sampling_ratio      = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
            pooler_type         = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

            boxfix_kernel = cfg.MODEL.ROI_MASK_HEAD.BOXFIX_KERNEL
            if boxfix_kernel == 3:
                padding_size = 1
            elif boxfix_kernel == 5:
                padding_size = 2

            if self.box_refine_on_mask_new_pooler is not None:
                self.refined_box_pooler = ROIPooler(
                    # output_size=pooler_resolution * 4, # 7x7 -> 28x28
                    output_size=self.mask_refine_size,
                    scales=pooler_scales,
                    sampling_ratio=sampling_ratio,
                    pooler_type=pooler_type,
                )
                if self.mask_refine_size == 14 and self.boundary_preserving_on:
                    self.refined_box_pooler_bp = ROIPooler(
                        output_size=pooler_resolution * 4, # 7x7 -> 28x28
                        # output_size=self.mask_refine_size,
                        scales=pooler_scales,
                        sampling_ratio=sampling_ratio,
                        pooler_type=pooler_type,
                    )

            # Box2BoxTransform for bounding box regression
            self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

            self.conv_norm_relus_box_fusion_list = []
            self.fcs_refine_list = []
            self.box_refine_predictor_list = []
            for mask_refine_idx in range(self.mask_refine_num):
                conv_norm_relus_box_fusion = []
                if self.mask_eee_on:
                    if self.error_estimation_class_agnostic:
                        box_fusion_channel = input_channels
                        if "feat" in self.fusion_targets:
                            box_fusion_channel += see_conv_dims
                        if "pred" in self.fusion_targets:
                            box_fusion_channel += error_dim
                        if self.dense_fusion_on:
                            box_fusion_channel += conv_dims
                            if self.fusion_pred_class_only or self.cls_agnostic_initial_mask:
                                box_fusion_channel += 1
                            else:
                                box_fusion_channel += num_classes
                    else:
                        box_fusion_channel = input_channels
                        if "feat" in self.fusion_targets:
                            box_fusion_channel += see_conv_dims
                        if "pred" in self.fusion_targets:
                            if self.error_estimation_fusion_class_agnostic:
                                box_fusion_channel += error_dim
                            else:
                                box_fusion_channel += error_dim * num_classes
                else:
                    box_fusion_channel = input_channels
                    # if "feat" in self.fusion_targets:
                    #     box_fusion_channel += conv_dims
                    # if "pred" in self.fusion_targets:
                    #     box_fusion_channel += num_classes
                
                
                for k in range(box_num_fusion_conv):
                    if self.fusion_with_depthwise_conv and k == 0:
                        conv = DepthwiseConv(
                            box_fusion_channel if k == 0 else boxfix_conv_dims,
                            boxfix_conv_dims,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=not self.norm,
                            norm=get_norm(self.norm, boxfix_conv_dims),
                            activation=F.relu,
                        )
                    else:
                        conv = Conv2d(
                                box_fusion_channel if k == 0 else boxfix_conv_dims,
                                boxfix_conv_dims,
                                kernel_size=boxfix_kernel,
                                stride=2,
                                padding=padding_size,
                                bias=not self.norm,
                                norm=get_norm(self.norm, boxfix_conv_dims),
                                activation=F.relu,
                            )
                    # box_fusion_channel = box_conv_dim
                    if mask_refine_idx == 0:
                        self.add_module("box_conv_refine{}".format(k + 1), conv)
                    else:
                        self.add_module("box_conv_refine{}_{}".format(mask_refine_idx, k + 1), conv)
                    conv_norm_relus_box_fusion.append(conv)
                self.conv_norm_relus_box_fusion_list.append(conv_norm_relus_box_fusion)
                feat_hw = int(np.round(self.mask_refine_size / (2 ** box_num_fusion_conv)))
                self._output_size_refine = (boxfix_conv_dims, feat_hw, feat_hw)
                for layer in conv_norm_relus_box_fusion:
                    if type(layer) == DepthwiseConv:
                        weight_init.c2_msra_fill(layer.depthwise)
                        weight_init.c2_msra_fill(layer.pointwise)
                    else:
                        weight_init.c2_msra_fill(layer)
                
                fcs_refine = []
                for k in range(boxfix_num_fc):
                    if k == 0:
                        input_fc_dim = np.prod(self._output_size_refine)
                        fc = nn.Linear(input_fc_dim, fc_dim)
                    else:
                        fc = nn.Linear(fc_dim, fc_dim)
                    if mask_refine_idx == 0:
                        self.add_module("fc_refine{}".format(k + 1), fc)
                    else:
                        self.add_module("fc_refine{}_{}".format(mask_refine_idx, k + 1), fc)
                    # self.add_module("fc_refine{}".format(k + 1), fc)
                    fcs_refine.append(fc)
                    self._output_size_refine = fc_dim
                self.fcs_refine_list.append(fcs_refine)
                
                for layer in fcs_refine:
                    weight_init.c2_xavier_fill(layer)
                
                if self.box_refine_on_mask_head_box_class_agnostic:
                    box_refine_predictor = FastRCNNOutputLayers(
                        self._output_size_refine, num_classes, self.box_refine_on_mask_head_box_class_agnostic, class_refine=self.box_refine_on_mask_head_class_refine
                    )
                else:
                    box_refine_predictor = FastRCNNOutputLayers(
                        self._output_size_refine, num_classes, self.cls_agnostic_bbox_reg, class_refine=self.box_refine_on_mask_head_class_refine
                    )
                if self.mask_refine_num == 1:
                    self.box_refine_predictor = box_refine_predictor
                else:
                    self.box_refine_predictor_list.append(box_refine_predictor)
                    if mask_refine_idx == 0:
                        self.add_module("box_refine_predictor", box_refine_predictor)
                    else:
                        self.add_module("box_refine_predictor_{}".format(mask_refine_idx), box_refine_predictor)

        if self.mask_refine_num > 1 and self.use_cascade_iou_thresh:
            self.num_classes = num_classes
            self.proposal_matchers = []
            cascade_ious = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
            for k in range(self.mask_refine_num+1):
                if k == 0:
                    # The first matching is done by the matcher of ROIHeads (self.proposal_matcher).
                    self.proposal_matchers.append(None)
                else:
                    self.proposal_matchers.append(
                        Matcher([cascade_ious[k]], [0, 1], allow_low_quality_matches=False)
                    )

        # Additional Layers from Other Models (e.g. RefineMask, PatchDCT)
        if self.mask_dct_on or self.mask_dct_after_maskfix_on:
            dct_vector_dim = 300
            # mask_size = self.mask_refine_size
            dct_loss_type = 'l1'
            mask_loss_para = 1.0
            self.dct_encoding = DctMaskEncoding(vec_dim=dct_vector_dim, mask_size=128)

            if self.mask_dct_on:
                self.predictor_fc1 = nn.Linear(256*14*14, 1024)
            elif self.mask_dct_after_maskfix_on:
                if self.deconv_after_maskfix:
                    self.predictor_fc1 = nn.Linear(256*28*28, 1024)
                else:
                    self.predictor_fc1 = nn.Linear(256*14*14, 1024)
            self.predictor_fc2 = nn.Linear(1024, 1024)
            self.predictor_fc3 = nn.Linear(1024, dct_vector_dim)

            for layer in [self.predictor_fc1, self.predictor_fc2]:
                weight_init.c2_xavier_fill(layer)
            
            nn.init.normal_(self.predictor_fc3.weight, std=0.001)
            nn.init.constant_(self.predictor_fc3.bias, 0)

        if self.boundary_preserving_on:
            if self.mask_initial_on and self.mask_refine_on:
                self.mask_final_fusion = Conv2d(
                    maskfix_conv_dims, maskfix_conv_dims,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=not self.norm,
                    norm=get_norm(self.norm, maskfix_conv_dims),
                    activation=F.relu)
                
                self.downsample = Conv2d(
                    maskfix_conv_dims, maskfix_conv_dims,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=not self.norm,
                    norm=get_norm(self.norm, maskfix_conv_dims),
                    activation=F.relu
                )
                num_boundary_conv = 2
                self.boundary_fcns = []
                cur_channels = input_shape.channels
                for k in range(num_boundary_conv):
                    conv = Conv2d(
                        cur_channels,
                        maskfix_conv_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not self.norm,
                        norm=get_norm(self.norm, maskfix_conv_dims),
                        activation=F.relu,
                    )
                    self.add_module("boundary_fcn{}".format(k + 1), conv)
                    self.boundary_fcns.append(conv)
                    cur_channels = maskfix_conv_dims

                self.mask_to_boundary = Conv2d(
                    maskfix_conv_dims, maskfix_conv_dims,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=not self.norm,
                    norm=get_norm(self.norm, maskfix_conv_dims),
                    activation=F.relu
                )

                self.boundary_to_mask = Conv2d(
                    maskfix_conv_dims, maskfix_conv_dims,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=not self.norm,
                    norm=get_norm(self.norm, maskfix_conv_dims),
                    activation=F.relu
                )

                # self.mask_deconv = ConvTranspose2d(
                #     conv_dims, conv_dims, kernel_size=2, stride=2, padding=0
                # )
                # self.mask_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

                self.boundary_deconv = ConvTranspose2d(
                    maskfix_conv_dims, maskfix_conv_dims, kernel_size=2, stride=2, padding=0
                )
                self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

                for layer in self.boundary_fcns +\
                            [self.boundary_deconv, self.boundary_to_mask, self.mask_to_boundary,
                            self.mask_final_fusion, self.downsample]:
                    weight_init.c2_msra_fill(layer)
                # use normal distribution initialization for mask prediction layer
                nn.init.normal_(self.boundary_predictor.weight, std=0.001)
                if self.boundary_predictor.bias is not None:
                    nn.init.constant_(self.boundary_predictor.bias, 0)
            else:
                conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
                num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
                conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
                # num_boundary_conv = cfg.MODEL.BOUNDARY_MASK_HEAD.NUM_CONV
                num_boundary_conv = 2
                num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
                if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
                    num_classes = 1
                
                self.mask_fcns = []
                cur_channels = input_shape.channels
                for k in range(num_conv):
                    conv = Conv2d(
                        cur_channels,
                        conv_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not conv_norm,
                        norm=get_norm(conv_norm, conv_dim),
                        activation=F.relu,
                    )
                    self.add_module("mask_fcn{}".format(k + 1), conv)
                    self.mask_fcns.append(conv)
                    cur_channels = conv_dim

                self.mask_final_fusion = Conv2d(
                    conv_dim, conv_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=not conv_norm,
                    norm=get_norm(conv_norm, conv_dim),
                    activation=F.relu)

                self.downsample = Conv2d(
                    conv_dim, conv_dim,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=not conv_norm,
                    norm=get_norm(conv_norm, conv_dim),
                    activation=F.relu
                )
                self.boundary_fcns = []
                cur_channels = input_shape.channels
                for k in range(num_boundary_conv):
                    conv = Conv2d(
                        cur_channels,
                        conv_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not conv_norm,
                        norm=get_norm(conv_norm, conv_dim),
                        activation=F.relu,
                    )
                    self.add_module("boundary_fcn{}".format(k + 1), conv)
                    self.boundary_fcns.append(conv)
                    cur_channels = conv_dim

                self.mask_to_boundary = Conv2d(
                    conv_dim, conv_dim,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=not conv_norm,
                    norm=get_norm(conv_norm, conv_dim),
                    activation=F.relu
                )

                self.boundary_to_mask = Conv2d(
                    conv_dim, conv_dim,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=not conv_norm,
                    norm=get_norm(conv_norm, conv_dim),
                    activation=F.relu
                )

                self.mask_deconv = ConvTranspose2d(
                    conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
                )
                self.mask_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

                self.boundary_deconv = ConvTranspose2d(
                    conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
                )
                self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

                for layer in self.mask_fcns + self.boundary_fcns +\
                            [self.mask_deconv, self.boundary_deconv, self.boundary_to_mask, self.mask_to_boundary,
                            self.mask_final_fusion, self.downsample]:
                    weight_init.c2_msra_fill(layer)
                # use normal distribution initialization for mask prediction layer
                nn.init.normal_(self.mask_predictor.weight, std=0.001)
                nn.init.normal_(self.boundary_predictor.weight, std=0.001)
                if self.mask_predictor.bias is not None:
                    nn.init.constant_(self.mask_predictor.bias, 0)
                if self.boundary_predictor.bias is not None:
                    nn.init.constant_(self.boundary_predictor.bias, 0)

        if self.refinemask_on:
            self.num_convs_instance = 2
            self.conv_kernel_size_instance = 3
            self.conv_in_channels_instance = 256
            self.conv_out_channels_instance = 256

            self.num_convs_semantic = 4
            self.conv_kernel_size_semantic = 3
            self.conv_in_channels_semantic = 256
            self.conv_out_channels_semantic = 256
            semantic_out_stride = 4

            self.conv_cfg = None
            self.norm_cfg = None

            self.semantic_out_stride = 4
            self.stage_sup_size = [14, 28, 56, 112]
            self.stage_num_classes = [80, 80, 80, 80]
            self.pre_upsample_last_stage = False

            fusion_type = 'MultiBranchFusion'
            dilations = [1, 3, 5]
            upsample_cfg = dict(type='bilinear', scale_factor=2)
            loss_cfg=dict(
                type='BARCrossEntropyLoss',
                stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                boundary_width=2,
                start_stage=1)
            
            if not self.mask_refine_on:
                self.refinemask_build_conv_layer('instance')
            self.refinemask_build_conv_layer('semantic')
            self.refinemask_loss_func = BARCrossEntropyLoss(
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    boundary_width=2,
                    start_stage=1)

            self.stages = nn.ModuleList()
            out_channel = self.conv_out_channels_instance
            stage_out_channels = [self.conv_out_channels_instance]
            for idx, out_size in enumerate(self.stage_sup_size[:-1]):
                in_channel = out_channel
                out_channel = in_channel // 2

                new_stage = SimpleSFMStage(
                    semantic_in_channel=self.conv_out_channels_semantic,
                    semantic_out_channel=in_channel,
                    instance_in_channel=in_channel,
                    instance_out_channel=out_channel,
                    fusion_type=fusion_type,
                    dilations=dilations,
                    out_size=out_size,
                    num_classes=self.stage_num_classes[idx],
                    semantic_out_stride=semantic_out_stride,
                    upsample_cfg=upsample_cfg)

                self.stages.append(new_stage)
                stage_out_channels.append(out_channel)

            self.stage_instance_logits = nn.ModuleList([
                nn.Conv2d(stage_out_channels[idx], num_classes, 1) for idx, num_classes in enumerate(self.stage_num_classes)])
            self.relu = nn.ReLU(inplace=True)

        if self.patchdct_on:
            # fmt: off
            # in_features       = cfg.MODEL.ROI_MASK_HEAD.FINE_FEATURES
            in_features       = ("p2", )
            #print('forcing use P2 as mask InFeatures')
            # pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.FINE_FEATURES_RESOLUTION
            pooler_resolution = 42
            # pooler_scales     = tuple(1.0 / self.feature_strides[k].stride for k in in_features)
            self.feature_strides = {k: v.stride for k, v in input_shape_pooler.items()}
            pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in in_features)
            sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
            pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
            # fmt: on
            # fine_features = cfg.MODEL.ROI_MASK_HEAD.FINE_FEATURES
            self.fine_features = in_features
            # in_channels = [input_shape[f].channels for f in in_features][0]

            # ret = {"fine_features":fine_features}
            self.fine_mask_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            
            self.patch_dct_vector_dim = 6
            self.mask_size_assemble = 112
            self.patch_size = 8
            self.num_classes = num_classes
            self.hidden_features = 1024
            self.dct_vector_dim = 300
            self.mask_size = 112
            self.dct_loss_type = "l1"
            self.mask_loss_para = 1.0
            self.scale = self.mask_size // self.patch_size
            self.ratio = 42 // self.scale
            self.patch_threshold = 0.30
            self.eval_gt = False
            self.num_stage = 2 - 1
            self.loss_para = [1.0, 1.0]
            # print("num stage of the model is {}".format(self.num_stage))

            self.dct_encoding = DctMaskEncoding(vec_dim=self.dct_vector_dim, mask_size=self.mask_size)
            self.patch_dct_encoding = DctMaskEncoding(vec_dim=self.patch_dct_vector_dim, mask_size=self.patch_size)
            self.gt = GT_infomation(self.mask_size_assemble, self.mask_size, self.patch_size, self.scale,
                                    self.dct_encoding, self.patch_dct_encoding)
            
            if not self.mask_initial_on:
                self.patchdct_conv_norm_relus = []

                cur_channels = input_shape.channels
                # for k, conv_dim in enumerate(patchdct_conv_dims[:-1]):
                for k in range(num_conv):
                    conv = Conv2d(
                        cur_channels,
                        # conv_dim,
                        conv_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        # bias=not conv_norm,
                        # norm=get_norm(conv_norm, conv_dim),
                        bias=not self.norm,
                        norm=get_norm(self.norm, conv_dims),
                        activation=F.relu,
                    )
                    self.add_module("patchdct_mask_fcn{}".format(k + 1), conv)
                    self.patchdct_conv_norm_relus.append(conv)
                for layer in self.patchdct_conv_norm_relus:
                    weight_init.c2_msra_fill(layer)
            
            if self.mask_refine_on:
                cur_channels = maskfix_conv_dims
                conv_dim = maskfix_conv_dims
            else:
                cur_channels = conv_dims
                conv_dim = conv_dims

            self.patchdct_predictor = nn.Sequential(
                nn.Linear(14 ** 2 * conv_dim, self.hidden_features),
                nn.ReLU(),
                nn.Linear(self.hidden_features, self.hidden_features),
                nn.ReLU(),
                nn.Linear(self.hidden_features, self.dct_vector_dim)
            )
            self.patchdct_reshape = Conv2d(
                1,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=F.relu
            )
            self.patchdct_fusion = nn.Sequential(
                Conv2d(cur_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    # bias=not conv_norm,
                    # norm=get_norm(conv_norm, conv_dim),
                    bias=not self.norm,
                    norm=get_norm(self.norm, conv_dim),
                    activation=F.relu),
                Conv2d(cur_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    # bias=not conv_norm,
                    # norm=get_norm(conv_norm, conv_dim),
                    bias=not self.norm,
                    norm=get_norm(self.norm, conv_dim),
                    activation=F.relu)
            )

            self.patchdct_downsample = nn.Sequential(
                Conv2d(
                    cur_channels,
                    self.hidden_features,
                    kernel_size=self.ratio,
                    stride=self.ratio,
                    padding=0,
                    # bias=not conv_norm,
                    # norm=get_norm(conv_norm, conv_dim),
                    bias=not self.norm,
                    norm=get_norm(self.norm, conv_dim),
                    activation=F.relu, ),
                Conv2d(self.hidden_features,
                    self.hidden_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    # bias=not conv_norm,
                    # norm=get_norm(conv_norm, conv_dim),
                    bias=not self.norm,
                    norm=get_norm(self.norm, conv_dim),
                    activation=F.relu),
            )

            self.patchdct_predictor1 = Conv2d(self.hidden_features,
                                    self.patch_dct_vector_dim * self.num_classes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    )
            self.patchdct_predictor_bfg = Conv2d(self.hidden_features,
                                        3 * self.num_classes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        )

        if self.transfiner_on:
            
            if self.mask_refine_on:
                cur_channels = maskfix_conv_dims
                conv_dim = maskfix_conv_dims
            else:
                cur_channels = conv_dims
                conv_dim = conv_dims
            self.conv_norm_relus_uncertain = []
            
            if not self.mask_initial_on:
                self.conv_norm_relus = []
                # cur_channels = input_shape.channels
                for k, conv_dim in enumerate(conv_dims[:-1]):
                    conv = Conv2d(
                        cur_channels,
                        conv_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not self.norm,
                        norm=get_norm(self.norm, conv_dim),
                        activation=nn.ReLU(),
                    )
                    self.add_module("mask_fcn{}".format(k + 1), conv)
                    self.conv_norm_relus.append(conv)
                    cur_channels = conv_dim

            self.deconv = ConvTranspose2d(
                cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
            )
            self.deconv_bo = ConvTranspose2d(
                cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
            )

            self.add_module("deconv_relu", nn.ReLU())
            cur_channels = conv_dims[-1]

            self.predictor = Conv2d(cur_channels, num_classes,
                                    kernel_size=1, stride=1, padding=0)
            self.predictor_bo = Conv2d(cur_channels, 1,
                                    kernel_size=1, stride=1, padding=0)

            encoder_layer = TransformerEncoderLayer(d_model=256, nhead=4)
            # used for the b4 and b4 correct; nice_light
            self.encoder = TransformerEncoder(encoder_layer, num_layers=3)
            for k, conv_dim in enumerate(conv_dims[:-1]):
                
                if k == 3:
                    conv_dim = 128
                conv = Conv2d(
                    cur_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not self.norm,
                    norm=get_norm(self.norm, conv_dim),
                    activation=nn.ReLU(),
                )
                self.add_module("mask_fcn_uncertain{}".format(k + 1), conv)
                self.conv_norm_relus_uncertain.append(conv)
                cur_channels = conv_dim

            self.deconv_uncertain = ConvTranspose2d(
                cur_channels, cur_channels, kernel_size=2, stride=2, padding=0
            )

            self.predictor_uncertain = Conv2d(cur_channels, 1,
                                            kernel_size=1, stride=1, padding=0)
            self.predictor_semantic_s = Conv2d(256, 1,
                                            kernel_size=1, stride=1, padding=0) # additional

            self.sig = nn.Sigmoid()

            for layer in self.conv_norm_relus + [self.deconv] + [self.deconv_bo] + self.conv_norm_relus_uncertain + [self.deconv_uncertain]:
                weight_init.c2_msra_fill(layer)
            # use normal distribution initialization for mask prediction layer
            nn.init.normal_(self.predictor.weight, std=0.001)
            nn.init.normal_(self.predictor_uncertain.weight, std=0.001)
            nn.init.normal_(self.predictor_bo.weight, std=0.001)
            if self.predictor.bias is not None:
                nn.init.constant_(self.predictor.bias, 0)
                nn.init.constant_(self.predictor_uncertain.bias, 0)
                nn.init.constant_(self.predictor_bo.bias, 0)


    def refinemask_build_conv_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = Conv2d(in_channels, out_channels, conv_kernel_size, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def refinemask_get_targets(self, pred_mask_logits, instances, pred_instances=None):

        mask_ratios = []

        # stage_instance_targets_list = [[] for _ in range(len(self.stage_sup_size))]
        stage_instance_targets_list = []
        for pred_mask_logit in pred_mask_logits:
            
            mask_side_len = pred_mask_logit.size(2)
            gt_masks = []
            gt_classes = []
            for idx, instances_per_image in enumerate(instances):
                if len(instances_per_image) == 0:
                    continue
                
                # if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)
                
                if pred_instances is None:
                    gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                        instances_per_image.proposal_boxes.tensor, mask_side_len
                    ).to(device=pred_mask_logit.device)
                    
                else:
                    gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                        pred_instances[idx].pred_boxes.tensor, mask_side_len
                    ).to(device=pred_mask_logit.device)
                
                gt_masks.append(gt_masks_per_image)
            stage_instance_targets_list.append(gt_masks)
        stage_instance_targets = [torch.cat(targets) for targets in stage_instance_targets_list]

        return stage_instance_targets

    def refinemask_get_seg_masks(self, mask_refine_logits, pred_instances=None, instances=None):
        stage_instance_preds = mask_refine_logits[1:]
        for idx in range(len(stage_instance_preds) - 1):
            instance_pred = stage_instance_preds[idx].squeeze(1).sigmoid() >= 0.5
            non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
            non_boundary_mask = F.interpolate(
                non_boundary_mask.float(),
                stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
            pre_pred = F.interpolate(
                stage_instance_preds[idx],
                stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
            stage_instance_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
        instance_pred = stage_instance_preds[-1]
        return instance_pred

    def stage_patch2mask(self, bfg, patch_vectors):
        device = bfg.device
        index = torch.argmax(bfg, dim=1)
        bg = torch.zeros_like(patch_vectors, device=device)
        bg[index == 1] = 1
        fg = torch.zeros_like(patch_vectors, device=device)
        fg[index == 2, 0] = self.patch_size
        masks = patch_vectors * bg + fg
        masks = self.patch_dct_encoding.decode(masks).real
        masks = patch2masks(masks, self.scale, self.patch_size, self.mask_size_assemble)
        return masks[:, None, :, :]

    def patchdct(self, masks, fine_mask_features):
        """
        PatchDCT block
        Args:
            fine_mask_features: feature map cropped from FPN P2
            masks: masks to be refined

        Returns:
            bfg and patch_vector of each PatchDCT block
        """
        masks = F.interpolate(masks, size=(self.scale * self.ratio, self.scale * self.ratio))
        masks = self.patchdct_reshape(masks)
        fine_mask_features = masks + fine_mask_features
        fine_mask_features = self.patchdct_fusion(fine_mask_features)
        fine_mask_features = self.patchdct_downsample(fine_mask_features)
        patch_vectors = self.patchdct_predictor1(fine_mask_features)
        bfg = self.patchdct_predictor_bfg(fine_mask_features)
        bfg = bfg.reshape(-1, self.num_classes, 3, self.scale, self.scale)
        patch_vectors = patch_vectors.reshape(-1, self.num_classes, self.patch_dct_vector_dim, self.scale, self.scale)
        return bfg, patch_vectors

    def transfiner_layers(self, x_list):
        x = x_list[0]
        x_c = x.clone() 
        x_hr = x_list[1] 
        x_hr_l = x_list[2] 
        x_hr_ll = x_list[3] 
    
        x_p2_s = x_list[4] 
        B, C, H, W = x.size()
        x_uncertain = x.clone().detach() # whether to detach this one

        if not self.mask_initial_on:
            for cnt, layer in enumerate(self.conv_norm_relus):
                x = layer(x)

        x_uncertain += x
    
        for cnt, layer in enumerate(self.conv_norm_relus_uncertain):
            x_uncertain = layer(x_uncertain)

        x_bo = x.clone()

        x = F.relu(self.deconv(x))
        mask = self.predictor(x)

        x_uncertain = self.deconv_uncertain(x_uncertain)
        mask_uncertain = self.sig(self.predictor_uncertain(x_uncertain))

        bound = None
        if self.training:
            x_p2_s = self.predictor_semantic_s(x_p2_s) # additional
            x_bo = F.relu(self.deconv_bo(x_bo))
            bound = self.predictor_bo(x_bo)

        return mask, mask_uncertain, bound, x_hr, x_hr_l, x_hr_ll, x_c, x_p2_s, self.encoder

    def forward(self, x_s, x_l=None, x_xl=None, proposals=None, features=None, targets=None):
        # x_s: RoI features, 14x14
        # x_l: RoI features, 28x28
        # x_xl: RoI features, 56x56 

        # if x_l is None, use x_s on SEE, BOXFIX, MASKFIX

        # Initial Mask
        mask_pred_list = []
        eee_pred_list = []
        refine_pred = None
        outputs_box_refine_list = []
        pred_instances_list = []
        boundary_logits = None
        patchdct_logits = None
        proposals_list = []
        pca_feat_list = None

        for mask_refine_idx in range(self.mask_refine_num):
            
            if self.mask_initial_on:
                x_im = x_s
                image_sizes = [x.image_size for x in proposals]
                if mask_refine_idx > 0:
                    x_im = torch.cat([x_im, eee_feat, eee_pred], dim=1)

                conv_norm_relus = self.conv_norm_relus_list[mask_refine_idx]
                for layer in conv_norm_relus:
                    x_im = layer(x_im)
                
                if self.mask_dct_on:
                    x_im = torch.flatten(x_im, start_dim=1)
                    x_im = F.relu(self.predictor_fc1(x_im))
                    x_im = F.relu(self.predictor_fc2(x_im))
                    mask_pred = self.predictor_fc3(x_im)
                else:
                    predictor = self.predictor_list[mask_refine_idx]
                    if self.mask_refine_size == 14:
                        mask_feat = x_im
                        mask_pred = predictor(mask_feat)
                    else:
                        mask_feat = F.relu(self.deconv(x_im))
                        mask_pred = predictor(mask_feat)
                mask_pred_list.append(mask_pred)

                pca_feat_list = None

                # if not self.mask_refine_on:
                #     # # visualize it
                #     _mask_feat = x_im.detach().cpu().numpy()
                #     _mask_pred = mask_pred.detach().cpu().numpy() # [N, 128, 14, 14]
                #     # _eee_feat = eee_feat.detach().cpu().numpy()
                #     # _eee_pred = eee_pred.detach().cpu().numpy() # [N, 256, 28, 28]
                #     # _refine_feat = refine_feat.detach().cpu().numpy()
                #     # _refine_pred = refine_pred.detach().cpu().numpy() # [N, 256, 28, 28]
                #     roi_labels = cat([p.pred_classes for p in proposals], dim=0)

                #     from sklearn.decomposition import PCA
                #     from sklearn.preprocessing import MinMaxScaler
                #     scaler = MinMaxScaler(clip=False)

                #     # apply PCA
                #     def apply_pca(features):
                #         c, h, w = features.shape
                #         features = features.reshape(c, -1).swapaxes(0, 1)
                #         pca_channel = 3
                #         pca = PCA(n_components=pca_channel)
                #         pca.fit(features)
                #         pca_features = pca.transform(features)
                #         for i in range(pca_channel):
                #             pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())
                #         if pca_channel == 2:
                #             pca_features = np.stack([pca_features[:, 0], pca_features[:, 1], np.zeros_like(pca_features[:, 0])], axis=1)
                #         pca_features_rgb = pca_features.reshape(h, w, 3)

                #         return pca_features_rgb

                #     coarse_feat_pca_list = []
                #     eee_feat_pca_list = []
                #     refine_feat_pca_list = []
                #     for i in range(mask_feat.shape[0]):
                #         coarse_feat_ = _mask_feat[i] # [128, 14, 14]
                #         # coarse_pred_ = _mask_pred[i]
                #         # refine_feat_ = _refine_feat[i] # [256, 28, 28]
                #         # refine_pred_ = _refine_pred[i]

                #         # eee_feat_ = _eee_feat[i]
                #         # eee_pred_ = _eee_pred[i] # [2, 28, 28]

                #         coarse_feat_pca = apply_pca(coarse_feat_)
                #         # eee_feat_pca = apply_pca(eee_feat_)
                #         # refine_feat_pca = apply_pca(refine_feat_)

                #         coarse_feat_pca_list.append(coarse_feat_pca.transpose(2, 0, 1))
                #         # eee_feat_pca_list.append(eee_feat_pca.transpose(2, 0, 1))
                #         # refine_feat_pca_list.append(refine_feat_pca.transpose(2, 0, 1))

                #     if len(coarse_feat_pca_list) > 0:
                #         coarse_feat_pca = np.stack(coarse_feat_pca_list)
                #         # eee_feat_pca = np.stack(eee_feat_pca_list)
                #         # refine_feat_pca = np.stack(refine_feat_pca_list)
                #         pca_feat_list = [torch.tensor(coarse_feat_pca).to('cuda'), None, None]
                #     else:
                #         pca_feat_list = [None, None, None]

            
            if self.mask_eee_on:
                eee_pred_prev = mask_pred.detach() if "eee_pred" in self.stop_gradient else mask_pred
                eee_feat_prev = mask_feat.detach() if "eee_feat" in self.stop_gradient else mask_feat
                if not self.cls_agnostic_initial_mask:
                    eee_pred_prev = eee_pred_prev.softmax(dim=1)
                                    
                if self.mask_refine_size == 14:
                    eee_roi = x_s.detach() if "eee_roi" in self.stop_gradient else x_s
                else:
                    eee_roi = x_l.detach() if "eee_roi" in self.stop_gradient else x_l
                
                if "feat" in self.fusion_targets and "pred" not in self.fusion_targets:
                    if self.eee_fusion_type == 'cat':
                        eee_feat = torch.cat([eee_roi, eee_feat_prev], dim=1)
                    elif self.eee_fusion_type == 'add':
                        eee_feat = eee_roi + eee_feat_prev
                if "pred" in self.fusion_targets and "feat" not in self.fusion_targets:
                    if self.fusion_pred_class_only:
                        eee_pred_prev = eee_pred_prev[:,:1]
                    else:
                        eee_feat = torch.cat([eee_roi, eee_pred_prev], dim=1)
                if "feat" in self.fusion_targets and "pred" in self.fusion_targets:
                    if self.fusion_pred_class_only:
                        if proposals[0].has("gt_classes"):
                            for idx, x in enumerate(proposals):
                                if idx == 0:
                                    gt_classes_cat = x.gt_classes
                                else:
                                    gt_classes_cat = torch.cat([gt_classes_cat, x.gt_classes], dim=0)
                        else:
                            for idx, x in enumerate(proposals):
                                if idx == 0:
                                    gt_classes_cat = x.pred_classes
                                else:
                                    gt_classes_cat = torch.cat([gt_classes_cat, x.pred_classes], dim=0)

                        # eee_pred_prev = eee_pred_prev.softmax(dim=1)
                        for gt_idx, gt_class in enumerate(gt_classes_cat):
                            pred_logits_class = eee_pred_prev[gt_idx, gt_class].unsqueeze(0).unsqueeze(0)
                            if gt_idx == 0:
                                pred_logits_filter = pred_logits_class
                            else:
                                pred_logits_filter = torch.cat([pred_logits_filter, pred_logits_class], dim=0)
                        try:
                            eee_pred_prev = pred_logits_filter
                        except UnboundLocalError:
                            eee_pred_prev = eee_pred_prev[:,:1]
                    if self.eee_fusion_type == 'cat':
                        eee_feat = torch.cat([eee_roi, eee_feat_prev, eee_pred_prev], dim=1)
                    elif self.eee_fusion_type == 'add':
                        eee_feat = torch.cat([eee_roi + eee_feat_prev, eee_pred_prev], dim=1)

                conv_norm_relus_i2e_fusion = self.conv_norm_relus_i2e_fusion_list[mask_refine_idx]
                for layer in conv_norm_relus_i2e_fusion:
                    eee_feat = layer(eee_feat)
                conv_norm_relus_eee = self.conv_norm_relus_eee_list[mask_refine_idx]
                for idx, layer in enumerate(conv_norm_relus_eee):
                    eee_feat = layer(eee_feat)
                if self.mask_refine_num == 1:
                    eee_pred = self.predictor_eee(eee_feat)
                else:
                    predictor_eee = self.predictor_eee_list[mask_refine_idx]
                    eee_pred = predictor_eee(eee_feat)
                eee_pred_list.append(eee_pred)

                
            if self.mask_refine_on and not self.mask_eee_on:
                eee_feat = mask_feat
                eee_pred = mask_pred
            
            if self.box_refine_on_mask_head:

                if self.mask_refine_size == 14:
                    box_refine_roi = x_s
                else:
                    box_refine_roi = x_l

                if self.use_cascade_iou_thresh:
                    if mask_refine_idx == 0:
                        proposals_refine = copy.deepcopy(proposals)
                    else:
                        proposals_refine = self._create_proposals_from_boxes(
                            pred_boxes, image_sizes
                        )
                    if self.training:
                        proposals_refine, _, _, _ = self._match_and_label_boxes(proposals_refine, mask_refine_idx+1, targets, eee_feat, eee_pred, box_refine_roi)
                        proposals_refine, _, eee_feat, eee_pred, box_refine_roi = select_foreground_proposals(proposals_refine, self.num_classes, eee_feat=eee_feat, eee_pred=eee_pred, box_refine_roi=box_refine_roi)
                    else:
                        # copy proposals pred_classes to proposals_refine when evaluation
                        for idx, x in enumerate(proposals):
                            proposals_refine[idx].pred_classes = x.pred_classes
                else:
                    # print(proposals)
                    # proposals_refine = copy.deepcopy(proposals)
                    proposals_refine = proposals
                    if mask_refine_idx > 0:
                        if self.training:
                            for idx, x in enumerate(proposals_refine):
                                proposals_refine[idx].proposal_boxes = pred_boxes[idx]
                        else:
                            for idx, x in enumerate(proposals_refine):
                                proposals_refine[idx].pred_boxes = pred_boxes[idx]
                proposals_list.append(proposals_refine)
                
                box_refine_feat_prev = eee_feat
                box_refine_pred_prev = eee_pred
                
                
                if self.mask_eee_on:
                    if "feat" in self.fusion_targets and "pred" in self.fusion_targets:
                        if self.dense_fusion_on:
                            box_refine_feat_2d = torch.cat([box_refine_roi, box_refine_feat_prev, box_refine_pred_prev.softmax(dim=1), eee_feat_prev, eee_pred_prev], dim=1)
                        else:
                            box_refine_feat_2d = torch.cat([box_refine_roi, box_refine_feat_prev, box_refine_pred_prev.softmax(dim=1)], dim=1)
                    elif "feat" in self.fusion_targets:
                        box_refine_feat_2d = torch.cat([box_refine_roi, box_refine_feat_prev], dim=1)
                    elif "pred" in self.fusion_targets:
                        box_refine_feat_2d = torch.cat([box_refine_roi, box_refine_pred_prev.softmax(dim=1)], dim=1)
                else:
                    box_refine_feat_2d = box_refine_roi
                
                conv_norm_relus_box_fusion = self.conv_norm_relus_box_fusion_list[mask_refine_idx]
                for idx, layer in enumerate(conv_norm_relus_box_fusion):
                    box_refine_feat_2d = layer(box_refine_feat_2d)
                box_refine_feat = torch.flatten(box_refine_feat_2d, start_dim=1)
                
                ### BoxFix fc layers
                fcs_refine = self.fcs_refine_list[mask_refine_idx]
                for layer in fcs_refine:
                    box_refine_feat = layer(box_refine_feat)
                if self.mask_refine_num == 1:
                    box_scores, box_refine_proposal_deltas = self.box_refine_predictor(box_refine_feat)
                else:
                    box_refine_predictor = self.box_refine_predictor_list[mask_refine_idx]
                    box_scores, box_refine_proposal_deltas = box_refine_predictor(box_refine_feat)

                if type(self.box_refine_loss_weight) == list:
                    box_refine_loss_weight = self.box_refine_loss_weight[mask_refine_idx]
                else:
                    box_refine_loss_weight = self.box_refine_loss_weight
                
                # if mask_refine_idx == 0:
                #     outputs_box_refine = FastRCNNOutputs(
                #             self.box2box_transform,
                #             box_scores,
                #             box_refine_proposal_deltas,
                #             # box_refine_proposal_deltas,
                #             proposals,
                #             self.smooth_l1_beta,
                #             class_refine=self.box_refine_on_mask_head_class_refine,
                #             loss_weight=box_refine_loss_weight,
                #             cls_loss_weight=self.cls_refine_loss_weight,
                #             on_mask_head=True
                #         )
                # else:
                outputs_box_refine = FastRCNNOutputs(
                        self.box2box_transform,
                        box_scores,
                        box_refine_proposal_deltas,
                        # box_refine_proposal_deltas,
                        # proposals_list[mask_refine_idx-1],
                        proposals_list[mask_refine_idx],
                        self.smooth_l1_beta,
                        class_refine=self.box_refine_on_mask_head_class_refine,
                        loss_weight=box_refine_loss_weight,
                        cls_loss_weight=self.cls_refine_loss_weight,
                        on_mask_head=True
                    )

                pred_instances = outputs_box_refine.inference_for_refine_mask_head(
                        self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img, self.box_refine_on_mask_head_box_class_agnostic
                    )

                outputs_box_refine_list.append(outputs_box_refine)
                pred_instances_list.append(pred_instances)

                pred_boxes =[x.pred_boxes for x in pred_instances]
                refine_x_roi = self.refined_box_pooler(features, pred_boxes)
                if self.mask_refine_size == 14:
                    x_s = refine_x_roi
                    if self.boundary_preserving_on:
                        x_l = self.refined_box_pooler_bp(features, pred_boxes)
                else:
                    x_l = refine_x_roi

        if self.mask_refine_on:
            refine_feat_prev = eee_feat.detach() if "refine_feat" in self.stop_gradient else eee_feat
            refine_pred_prev = eee_pred.detach() if "refine_pred" in self.stop_gradient else eee_pred
            
            if self.deconv_before_maskfix:
                refine_roi = x_xl.detach() if "refine_feat" in self.stop_gradient else x_xl
                refine_feat_prev = F.relu(self.deconv_before_maskfix_layer(refine_feat_prev))
                refine_pred_prev = F.relu(self.deconv_before_maskfix_layer_pred(refine_pred_prev))
                if self.dense_fusion_on:
                    eee_feat_prev = F.relu(self.deconv_before_maskfix_layer_dense(eee_feat_prev))
                    eee_pred_prev = F.relu(self.deconv_before_maskfix_layer_dense_pred(eee_pred_prev))
            else:
                if self.mask_refine_size == 14:
                    refine_roi = x_s.detach() if "refine_roi" in self.stop_gradient else x_s
                else:
                    refine_roi = x_l.detach() if "refine_roi" in self.stop_gradient else x_l
            
            if self.dense_fusion_on:
                if "feat" in self.fusion_targets and "pred" not in self.fusion_targets:
                    refine_feat = torch.cat([refine_roi, eee_feat_prev, refine_feat_prev], dim=1)
                if "pred" in self.fusion_targets and "feat" not in self.fusion_targets:
                    refine_feat = torch.cat([refine_roi, eee_pred_prev, refine_pred_prev.softmax(dim=1)], dim=1)
                if "feat" in self.fusion_targets and "pred" in self.fusion_targets:
                    refine_feat = torch.cat([refine_roi, eee_feat_prev, eee_pred_prev, refine_feat_prev, refine_pred_prev.softmax(dim=1)], dim=1)
            elif self.error_fusion_maskfix:
                if "feat" in self.fusion_targets and "pred" not in self.fusion_targets:
                    if self.eee_fusion_type == 'cat':
                        refine_feat = torch.cat([refine_roi, refine_feat_prev], dim=1)
                    elif self.eee_fusion_type == 'add':
                        refine_feat = refine_roi + refine_feat_prev
                if "pred" in self.fusion_targets and "feat" not in self.fusion_targets:
                    refine_feat = torch.cat([refine_roi, refine_pred_prev.softmax(dim=1)], dim=1)
                if "feat" in self.fusion_targets and "pred" in self.fusion_targets:
                    if self.eee_fusion_type == 'cat':
                        refine_feat = torch.cat([refine_roi, refine_feat_prev, refine_pred_prev.softmax(dim=1)], dim=1)
                    elif self.eee_fusion_type == 'add':
                        refine_feat = torch.cat([refine_roi + refine_feat_prev, refine_pred_prev.softmax(dim=1)], dim=1)
            else:
                refine_feat = refine_roi

            conv_norm_relus_e2r_fusion = self.conv_norm_relus_e2r_fusion_list[0]
            for layer in conv_norm_relus_e2r_fusion:
                refine_feat = layer(refine_feat)
            # conv_norm_relus_refine = self.conv_norm_relus_refine_list[0]
            for idx, layer in enumerate(self.conv_norm_relus_refine):
                if self.maskfix_fusion_with_deform_conv and idx == 0:
                    offset = self.maskfix_deformable_offset(refine_feat)
                    refine_feat = F.relu(layer(refine_feat, offset))
                else:
                    refine_feat = layer(refine_feat)
            
            # DualFix + Boundary Preserving Module
            if self.boundary_preserving_on:
                boundary_features = self.downsample(x_l)
                boundary_features = boundary_features + self.mask_to_boundary(refine_feat)
                for layer in self.boundary_fcns:
                    boundary_features = layer(boundary_features)
                refine_feat = self.boundary_to_mask(boundary_features) + refine_feat
                refine_feat = self.mask_final_fusion(refine_feat)

                boundary_features = F.relu(self.boundary_deconv(boundary_features))
                boundary_logits = self.boundary_predictor(boundary_features)

            _refine_feat = refine_feat.detach().cpu().numpy() # shsh
            if self.deconv_after_maskfix:
                refine_feat = F.relu(self.deconv_after_maskfix_layer(refine_feat))
            
            # DualFix + DCT 
            if self.mask_dct_after_maskfix_on:
                refine_feat = torch.flatten(refine_feat, start_dim=1)
                refine_feat = F.relu(self.predictor_fc1(refine_feat))
                refine_feat = F.relu(self.predictor_fc2(refine_feat))
                refine_pred = self.predictor_fc3(refine_feat)
            
            elif not self.refinemask_on and not self.patchdct_on:
                refine_pred = self.predictor_refine(refine_feat)

            pca_feat_list = None
            # # visualize it
            # _mask_feat = mask_feat.detach().cpu().numpy()
            # _mask_pred = mask_pred.detach().cpu().numpy() # [N, 128, 14, 14]
            # _eee_feat = eee_feat.detach().cpu().numpy()
            # _eee_pred = eee_pred.detach().cpu().numpy() # [N, 256, 28, 28]
            # # _refine_feat = refine_feat.detach().cpu().numpy()
            # _refine_pred = refine_pred.detach().cpu().numpy() # [N, 256, 28, 28]
            # roi_labels = cat([p.pred_classes for p in proposals], dim=0)

            # from sklearn.decomposition import PCA
            # from sklearn.preprocessing import MinMaxScaler
            # scaler = MinMaxScaler(clip=False)

            # # apply PCA
            # def apply_pca(features):
            #     c, h, w = features.shape
            #     features = features.reshape(c, -1).swapaxes(0, 1)
            #     pca_channel = 3
            #     pca = PCA(n_components=pca_channel)
            #     pca.fit(features)
            #     pca_features = pca.transform(features)
            #     for i in range(pca_channel):
            #         pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())
            #     if pca_channel == 2:
            #         pca_features = np.stack([pca_features[:, 0], pca_features[:, 1], np.zeros_like(pca_features[:, 0])], axis=1)
            #     pca_features_rgb = pca_features.reshape(h, w, 3)

            #     return pca_features_rgb

            # coarse_feat_pca_list = []
            # eee_feat_pca_list = []
            # refine_feat_pca_list = []
            # for i in range(mask_feat.shape[0]):
            #     coarse_feat_ = _mask_feat[i] # [128, 14, 14]
            #     coarse_pred_ = _mask_pred[i]
            #     refine_feat_ = _refine_feat[i] # [256, 28, 28]
            #     refine_pred_ = _refine_pred[i]

            #     eee_feat_ = _eee_feat[i]
            #     eee_pred_ = _eee_pred[i] # [2, 28, 28]

            #     # import matplotlib.pyplot as plt
            #     # plt.subplot(3, 2, 1)
            #     # coarse_feat_pca = apply_pca(coarse_feat_)
            #     # plt.imshow(coarse_feat_pca)
            #     # plt.subplot(3, 2, 2)
            #     # plt.imshow(coarse_pred_[roi_labels[i]] > 0)
            #     # plt.subplot(3, 2, 3)
            #     # eee_feat_pca = apply_pca(eee_feat_)
            #     # plt.imshow(eee_feat_pca)
            #     # plt.subplot(3, 2, 4)
            #     # # visualize eee_pred with argmax
            #     # eee_pred_ = eee_pred_.argmax(0)
            #     # plt.imshow(eee_pred_)

            #     # plt.subplot(3, 2, 5)
            #     # refine_feat_pca = apply_pca(refine_feat_)
            #     # plt.imshow(refine_feat_pca)
            #     # plt.subplot(3, 2, 6)
            #     # plt.imshow(refine_pred_[roi_labels[i]] > 0)
            #     # plt.savefig('pca/{}.png'.format(i))

            #     coarse_feat_pca = apply_pca(coarse_feat_)
            #     eee_feat_pca = apply_pca(eee_feat_)
            #     refine_feat_pca = apply_pca(refine_feat_)

            #     coarse_feat_pca_list.append(coarse_feat_pca.transpose(2, 0, 1))
            #     eee_feat_pca_list.append(eee_feat_pca.transpose(2, 0, 1))
            #     refine_feat_pca_list.append(refine_feat_pca.transpose(2, 0, 1))

            # if len(coarse_feat_pca_list) > 0:
            #     coarse_feat_pca = np.stack(coarse_feat_pca_list)
            #     eee_feat_pca = np.stack(eee_feat_pca_list)
            #     refine_feat_pca = np.stack(refine_feat_pca_list)
            #     pca_feat_list = [torch.tensor(coarse_feat_pca).to('cuda'), torch.tensor(eee_feat_pca).to('cuda'), torch.tensor(refine_feat_pca).to('cuda')]
            # else:
            #     pca_feat_list = [None, None, None]

        # RefineMask only & DualFix + RefineMask
        if self.refinemask_on:

            if self.mask_initial_on:
                instance_feats = refine_feat
            else:
                instance_feats = x_s
            
            # Semantic Head use P2 feature of FPN
            semantic_feat = features[0]

            if not self.mask_refine_on:
                for conv in self.instance_convs:
                    instance_feats = conv(instance_feats)
            
            for conv in self.semantic_convs:
                semantic_feat = conv(semantic_feat)
            
            # rois = cat(pred_boxes, dim=0)
            if self.mask_initial_on:
                rois = pred_boxes
                if outputs_box_refine.has_gt:
                    roi_labels = cat([p.gt_classes for p in proposals], dim=0)
                else:
                    roi_labels = cat([p.pred_classes for p in proposals], dim=0)
            else:
                if proposals[0].has("proposal_boxes"):
                    rois = [p.proposal_boxes for p in proposals]
                else:
                    rois = [p.pred_boxes for p in proposals]
                try:
                    roi_labels = cat([p.gt_classes for p in proposals], dim=0)
                except:
                    roi_labels = cat([p.pred_classes for p in proposals], dim=0)
            # self.pred_classes = cat([p.pred_classes for p in proposals], dim=0)
            # print(roi_labels)

            stage_instance_preds = []
            for idx, stage in enumerate(self.stages):
                instance_logits = self.stage_instance_logits[idx](instance_feats)[torch.arange(len(roi_labels)), roi_labels][:, None]
                upsample_flag = self.pre_upsample_last_stage or idx < len(self.stages) - 1
                instance_feats = stage(instance_feats, instance_logits, semantic_feat, rois, upsample_flag)
                stage_instance_preds.append(instance_logits)

            # if use class-agnostic classifier for the last stage
            if self.stage_num_classes[-1] == 1:
                roi_labels = roi_labels.clamp(max=0)

            instance_preds = self.stage_instance_logits[-1](instance_feats)[torch.arange(len(roi_labels)), roi_labels][:, None]
            if not self.pre_upsample_last_stage:
                instance_preds = F.interpolate(instance_preds, scale_factor=2, mode='bilinear', align_corners=True)
            stage_instance_preds.append(instance_preds)

            if self.mask_refine_on:
                refine_pred = stage_instance_preds
            else:
                mask_pred = stage_instance_preds
                mask_pred_list.append(mask_pred)

        # DualFix + PatchDCT
        elif self.patchdct_on:
            
            # Alredy have refine_feat (processed by conv)
            if self.mask_refine_on:
                x = refine_feat
            else:
                x = x_s
                for layer in self.patchdct_conv_norm_relus:
                    x = layer(x)

            # DCT-Mask 
            x = self.patchdct_predictor(x.flatten(start_dim=1))
            if not self.training:
                num_masks = x.shape[0]
                if num_masks == 0:
                    # return x, 0, 0
                    if self.mask_refine_on:
                        refine_pred = x
                    else:
                        mask_pred = x
                        mask_pred_list.append(mask_pred)
                    patchdct_logits = [0, 0]
                    return mask_pred_list, eee_pred_list, refine_pred, outputs_box_refine_list, pred_instances_list, boundary_logits, patchdct_logits, proposals_list
            # reverse transform to obtain high-resolution masks
            # masks = self.dct_encoding.decode(x).real.reshape(-1, 1, self.mask_size, self.mask_size)
            masks = self.dct_encoding.decode(x).reshape(-1, 1, self.mask_size, self.mask_size)

            if self.box_refine_on_mask_head:
                pred_boxes =[x.pred_boxes for x in pred_instances]
                x_patchdct = self.fine_mask_pooler([features[0]], pred_boxes)
            else:
                if proposals[0].has("proposal_boxes"):
                    proposal_boxes = [p.proposal_boxes for p in proposals]
                else:
                    proposal_boxes = [p.pred_boxes for p in proposals]
                x_patchdct = self.fine_mask_pooler([features[0]], proposal_boxes)

            # PatchDCT
            bfg, patch_vectors = self.patchdct(masks, x_patchdct)

            if self.num_stage == 1:
                # return x, bfg, patch_vectors
                if self.mask_refine_on:
                    refine_pred = x
                else:
                    mask_pred = x
                    mask_pred_list.append(mask_pred)
                patchdct_logits = [bfg, patch_vectors]

            else:
                # for multi-stage PatchDCT
                if self.training:
                    classes = self.gt.get_gt_classes(pred_instances)
                else:
                    classes = pred_instances[0].pred_classes
                num_instance = classes.size()[0]
                indices = torch.arange(num_instance)
                bfg = bfg[indices, classes].permute(0, 2, 3, 1).reshape(-1, 3)
                patch_vectors = patch_vectors[indices, classes].permute(0, 2, 3, 1).reshape(-1, self.patch_dct_vector_dim)

                bfg_dict = {}
                patch_vectors_dict = {}
                bfg_dict[0] = bfg
                patch_vectors_dict[0] = patch_vectors
                for i in range(1, self.num_stage):
                    masks = self.stage_patch2mask(bfg, patch_vectors)
                    bfg, patch_vectors = self.patchdct(masks, x_patchdct)
                    bfg = bfg[indices, classes].permute(0, 2, 3, 1).reshape(-1, 3)
                    patch_vectors = patch_vectors[indices, classes].permute(0, 2, 3, 1).reshape(-1,
                                                                                                self.patch_dct_vector_dim)
                    bfg_dict[i] = bfg
                    patch_vectors_dict[i] = patch_vectors

        # DualRefine + Mask-Transfiner
        elif self.transfiner_on:
            
            if self.mask_refine_on:
                x = refine_feat
            else:
                x = x_s
            x, x_uncertain, x_bo, x_hr, x_hr_l, x_hr_ll, x_c, x_p2_s, encoder = self.transfiner_layers(x)

        # Boundary Preserving Mask R-CNN
        elif self.boundary_preserving_on:
            if not self.mask_initial_on:
                mask_features = x_s
                boundary_features = x_l
                for layer in self.mask_fcns:
                    mask_features = layer(mask_features)
                # downsample
                boundary_features = self.downsample(boundary_features)
                # mask to boundary fusion
                boundary_features = boundary_features + self.mask_to_boundary(mask_features)
                for layer in self.boundary_fcns:
                    boundary_features = layer(boundary_features)
                # boundary to mask fusion
                mask_features = self.boundary_to_mask(boundary_features) + mask_features
                mask_features = self.mask_final_fusion(mask_features)
                # mask prediction
                mask_features = F.relu(self.mask_deconv(mask_features))
                mask_logits = self.mask_predictor(mask_features)
                mask_pred_list.append(mask_logits)
                # boundary prediction
                boundary_features = F.relu(self.boundary_deconv(boundary_features))
                boundary_logits = self.boundary_predictor(boundary_features)

        return mask_pred_list, eee_pred_list, refine_pred, outputs_box_refine_list, pred_instances_list, boundary_logits, patchdct_logits, proposals_list, pca_feat_list

    def patchdct_mask_rcnn_dct_loss(self, pred_mask_logits, bfg, patch_vectors, instances, pred_instances=None, vis_period=0):
        """
        Compute the mask prediction loss defined in the Mask R-CNN paper.

        Args:
            pred_mask_logits (Tensor): [B, D]. D is dct-dim. [B, D]. DCT_Vector in DCT-Mask.
            bfg: A tensor of shape [B,num_class,3,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                A NxN masks is divided into scale x scale patches.
                bfg demonstrates results of three-class classifier in PatchDCT
                0 for foreground,1 for mixed,2 for background
            patch_vectors : A tensor of shape:[B,num_class,patch_dct_vector_dim,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                    DCT vector for each patch (only calculate loss for mixed patch)
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            vis_period (int): the period (in steps) to dump visualization.

        Returns:
            mask_loss (Tensor): A scalar tensor containing the loss.
        """

        if self.dct_loss_type == "l1":
            loss_func = F.l1_loss
        elif self.dct_loss_type == "sl1":
            loss_func = F.smooth_l1_loss
        elif self.dct_loss_type == "l2":
            loss_func = F.mse_loss
        else:
            raise ValueError("Loss Type Only Support : l1, l2; yours: {}".format(self.dct_loss_type))

        gt_masks, gt_classes, gt_masks_coarse, gt_bfg = self.gt.get_gt_mask(instances, pred_mask_logits, pred_instances=pred_instances)

        mask_loss = self.loss_para[0] * loss_func(pred_mask_logits, gt_masks_coarse)

        if self.num_stage == 1:

            num_instance = gt_classes.size()[0]
            indice = torch.arange(num_instance)
            bfg = bfg[indice, gt_classes].permute(0, 2, 3, 1).reshape(-1, 3)
            patch_vectors = patch_vectors[indice, gt_classes].permute(0, 2, 3, 1).reshape(-1, self.patch_dct_vector_dim)
            patch_vectors = patch_vectors[gt_bfg == 1, :]
            mask_loss_2 = F.cross_entropy(bfg, gt_bfg)
            mask_loss_3 = loss_func(patch_vectors, gt_masks)
            mask_loss = mask_loss + self.loss_para[1] * (mask_loss_2 + mask_loss_3)
            mask_loss = self.mask_loss_para * mask_loss
        else:
            for i in range(self.num_stage):
                bfg_this_stage = bfg[i]
                patch_vectors_this_stage = patch_vectors[i]
                patch_vectors_this_stage = patch_vectors_this_stage[gt_bfg == 1]
                mask_loss += self.loss_para[i + 1] * (
                            F.cross_entropy(bfg_this_stage, gt_bfg) + loss_func(patch_vectors_this_stage, gt_masks))
        return mask_loss

    def patchdct_mask_rcnn_dct_inference(self, pred_mask_logits, bfg, patch_vectors, pred_instances, instances=None):
        """
        Convert pred_mask_logits to estimated foreground probability masks while also
        extracting only the masks for the predicted classes in pred_instances. For each
        predicted box, the mask of the same class is attached to the instance by adding a
        new "pred_masks" field to pred_instances.

        Args:
            pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
            bfg: A tensor of shape [B,num_class,3,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                A NxN masks is divided into scale x scale patches.
                bfg demonstrates results of three-class classifier in PatchDCT
                0 for foreground,1 for mixed,2 for background
            patch_vectors : A tensor of shape:[B,num_class,patch_dct_vector_dim,scale,scale] or a dict of tensors(for multi-stage PatchDCT)
                    DCT vector for each patch (only calculate loss for mixed patch)
            pred_instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. Each Instances must have field "pred_classes".

        Returns:
            None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
                Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
                masks the resolution predicted by the network; post-processing steps, such as resizing
                the predicted masks to the original image resolution and/or binarizing them, is left
                to the caller.
        """

        num_patch = pred_mask_logits.shape[0]
        device = pred_mask_logits.device
        if num_patch == 0:
            # pred_instances[0].pred_masks = torch.empty([0, 1, self.mask_size, self.mask_size]).to(device)
            # return pred_instances
            return torch.empty([0, 1, self.mask_size, self.mask_size]).to(device)
        else:

            # pred_classes = pred_instances[0].pred_classes
            pred_classes = instances[0].pred_classes
            num_masks = pred_classes.shape[0]
            indices = torch.arange(num_masks)
            if self.num_stage > 1:
                bfg = bfg[self.num_stage - 1]
                patch_vectors = patch_vectors[self.num_stage - 1]
            else:
                bfg = bfg[indices, pred_classes].permute(0, 2, 3, 1).reshape(-1, 3)
                patch_vectors = patch_vectors[indices, pred_classes].permute(0, 2, 3, 1).reshape(-1,
                                                                                                 self.patch_dct_vector_dim)

            with torch.no_grad():

                bfg = F.softmax(bfg, dim=1)
                bfg[bfg[:, 0] > self.patch_threshold, 0] = bfg[bfg[:, 0] > self.patch_threshold, 0] + 1
                bfg[bfg[:, 2] > self.patch_threshold, 2] = bfg[bfg[:, 2] > self.patch_threshold, 2] + 1
                index = torch.argmax(bfg, dim=1)

                if self.eval_gt:
                    gt_masks, index = self.gt.get_gt_mask_inference(pred_instances, pred_mask_logits)
                    patch_vectors[index == 1] = gt_masks

                patch_vectors[index == 0, ::] = 0
                patch_vectors[index == 2, ::] = 0
                patch_vectors[index == 2, 0] = self.patch_size

                pred_mask_rc = self.patch_dct_encoding.decode(patch_vectors)
                # assemble patches to obtain an entire mask
                pred_mask_rc = patch2masks(pred_mask_rc, self.scale, self.patch_size, self.mask_size_assemble)

            pred_mask_rc = pred_mask_rc[:, None, :, :]
            # pred_instances[0].pred_masks = pred_mask_rc
            # return pred_instances
            return pred_mask_rc

    def mask_rcnn_dct_inference(self, pred_mask_logits, pred_instances, instances=None):
        """
        Convert pred_mask_logits to estimated foreground probability masks while also
        extracting only the masks for the predicted classes in pred_instances. For each
        predicted box, the mask of the same class is attached to the instance by adding a
        new "pred_masks" field to pred_instances.

        Args:
            pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
            pred_instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. Each Instances must have field "pred_classes".

        Returns:
            None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
                Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
                masks the resolution predicted by the network; post-processing steps, such as resizing
                the predicted masks to the original image resolution and/or binarizing them, is left
                to the caller.
        """
        num_masks = pred_mask_logits.shape[0]
        device = pred_mask_logits.device
        if num_masks == 0:
            # pred_instances[0].pred_masks = torch.empty([0, 1, self.mask_refine_size, self.mask_refine_size]).to(device)
            # return pred_instances
            pred_mask_rc = torch.empty([0, 1, self.mask_refine_size, self.mask_refine_size]).to(device)
            return pred_mask_rc
        else:
            pred_mask_rc = self.dct_encoding.decode(pred_mask_logits.detach())
            # print("=======shape+++++++++")
            # print(pred_mask_rc.shape)
            # pred_mask_rc = self.dct_encoding.decode(pred_mask_logits)
            pred_mask_rc = pred_mask_rc[:, None, :, :]
            # print(pred_mask_rc.shape)
            # print(torch.unique(pred_mask_rc))
            return pred_mask_rc
    
    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets, eee_feat, eee_pred, box_refine_roi):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        idx = 0
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
                gt_masks = targets_per_image.gt_masks[matched_idxs]

            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                gt_masks = BitMasks(
                    targets_per_image.gt_masks.tensor.new_zeros((len(proposals_per_image), 28, 28))
                )
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            proposals_per_image.gt_masks = gt_masks

            # start_idx = int(gt_classes.shape[0] * idx)
            # end_idx = int(gt_classes.shape[0] * (idx + 1))
            
            # # print("eee_pred", eee_pred.size())
            # eee_feat_per_image = eee_feat[start_idx:end_idx]
            # eee_pred_per_image = eee_pred[start_idx:end_idx]
            # box_refine_roi_per_image = box_refine_roi[start_idx:end_idx]
            # # print("eee_pred_per_image", eee_pred_per_image.size())
            # if eee_pred_per_image.shape[0] == 0:
            #     continue
            # if idx == 0:
            #     eee_feat_cat = eee_feat_per_image[matched_idxs]
            #     eee_pred_cat = eee_pred_per_image[matched_idxs]
            #     box_refine_roi_cat = box_refine_roi_per_image[matched_idxs]
            #     # print("eee_pred_cat", eee_pred_cat.size())
            # else:
            #     eee_feat_cat = torch.cat([eee_feat_cat, eee_feat_per_image[matched_idxs]], dim=0)
            #     eee_pred_cat = torch.cat([eee_pred_cat, eee_pred_per_image[matched_idxs]], dim=0)
            #     box_refine_roi_cat = torch.cat([box_refine_roi_cat, box_refine_roi_per_image[matched_idxs]], dim=0)

            idx += 1

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals, eee_feat, eee_pred, box_refine_roi

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        # boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            if self.training:
                prop.proposal_boxes = boxes_per_image
            else:
                prop.pred_boxes = boxes_per_image
            proposals.append(prop)
        return proposals

def build_mask_head(cfg, input_shape, input_shape_pooler):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape, input_shape_pooler)

