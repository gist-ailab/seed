# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F
from mask_eee_rcnn.layers import paste_masks_in_image
from mask_eee_rcnn.structures import Instances
import copy

def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    
    if results.has("pred_boxes_initial"):
        results.pred_boxes_initial.scale(scale_x, scale_y)
        results.pred_boxes_initial.clip(results.image_size)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = paste_masks_in_image(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has('pred_masks_initial'):
        if results.has('pred_boxes_initial'):
            results.pred_masks_initial = paste_masks_in_image(
                results.pred_masks_initial[:, 0, :, :],  # N, 1, M, M
                results.pred_boxes_initial,
                results.image_size,
                threshold=mask_threshold,
            )
        else:
            results.pred_masks_initial = paste_masks_in_image(
                results.pred_masks_initial[:, 0, :, :],  # N, 1, M, M
                results.pred_boxes,
                results.image_size,
                threshold=mask_threshold,
            )
    
    if results.has('pred_errors'):
        if len(torch.unique(results.pred_errors)) <= 2:
            results.pred_errors = paste_masks_in_image(
                results.pred_errors == 1,
                results.pred_boxes_initial,
                results.image_size,
                threshold=mask_threshold,
            )
        else:
            if results.has('pred_boxes_initial'):
                results.pred_errors = paste_masks_in_image(
                    # copy.deepcopy(results.pred_errors) == 2,
                    results.pred_errors,
                    results.pred_boxes_initial,
                    results.image_size,
                    # threshold=mask_threshold,
                    threshold=-1,
                    # mode="nearest",
                )
            else:
                results.pred_errors = paste_masks_in_image(
                    # copy.deepcopy(results.pred_errors) == 2,
                    results.pred_errors,
                    results.pred_boxes,
                    results.image_size,
                    # threshold=mask_threshold,
                    threshold=-1,
                    # mode="nearest",
                )
            results.pred_negative_errors = results.pred_errors == 3

    
    if results.has('coarse_feat_pca'):
        if results.has('pred_boxes_initial'):
            # print("course feat shape", results.coarse_feat_pca.shape)
            coarse_feat_pca_list = []
            for i in range(results.coarse_feat_pca.shape[1]):
                coarse_feat_pca = paste_masks_in_image(
                    results.coarse_feat_pca[:, i, :, :],
                    results.pred_boxes_initial,
                    results.image_size,
                    threshold=-1,
                )
                coarse_feat_pca_list.append(coarse_feat_pca)
            results.coarse_feat_pca = torch.stack(coarse_feat_pca_list, dim=1)
        else:
            coarse_feat_pca_list = []
            for i in range(results.coarse_feat_pca.shape[1]):
                coarse_feat_pca = paste_masks_in_image(
                    results.coarse_feat_pca[:, i, :, :],
                    results.pred_boxes,
                    results.image_size,
                    threshold=-1,
                )
                coarse_feat_pca_list.append(coarse_feat_pca)
            results.coarse_feat_pca = torch.stack(coarse_feat_pca_list, dim=1)

    if results.has('eee_feat_pca'):
        eee_feat_pca_list = []
        for i in range(results.coarse_feat_pca.shape[1]):
            eee_feat_pca = paste_masks_in_image(
                results.eee_feat_pca[:, i, :, :],
                results.pred_boxes,
                results.image_size,
                threshold=-1,
            )
            eee_feat_pca_list.append(eee_feat_pca)
        results.eee_feat_pca = torch.stack(eee_feat_pca_list, dim=1)
    
    if results.has('refine_feat_pca'):
        refine_feat_pca_list = []
        for i in range(results.coarse_feat_pca.shape[1]):
            refine_feat_pca = paste_masks_in_image(
                results.refine_feat_pca[:, i, :, :],
                results.pred_boxes,
                results.image_size,
                threshold=-1,
            )
            refine_feat_pca_list.append(refine_feat_pca)
        results.refine_feat_pca = torch.stack(refine_feat_pca_list, dim=1)

    if results.has('gt_errors'):
        gt_crrors_list = []
        results.gt_errors = paste_masks_in_image(
            results.gt_errors,
            results.pred_boxes_initial,
            results.image_size,
            threshold=-1,
            # mode="nearest",
        )

    if results.has('pred_var'):
        results.pred_var = paste_masks_in_image(
            results.pred_var[:, 0, :, :],
            results.pred_boxes,
            results.image_size,
            threshold=-1,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result
