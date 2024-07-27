# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import cv2
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

from mask_eee_rcnn.layers import ShapeSpec
from mask_eee_rcnn.structures import Boxes, Instances, pairwise_iou
from mask_eee_rcnn.utils.events import get_event_storage
from mask_eee_rcnn.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, make_stage
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, FastRCNNEEOutputs
from .keypoint_head import build_keypoint_head, keypoint_rcnn_inference, keypoint_rcnn_loss
from .mask_head import build_mask_head, mask_rcnn_inference, mask_rcnn_loss, mask_rcnn_eee_loss, mask_rcnn_dct_loss, boundary_preserving_mask_loss
from .maskiou_head import build_maskiou_head, mask_iou_inference, mask_iou_loss
from .mask_eee_head import build_mask_eee_head, mask_eee_inference, mask_eee_loss
from .fast_rcnn import fast_rcnn_inference_refine
from .refinemask_loss import BARCrossEntropyLoss

from fvcore.nn.parameter_count import parameter_count_table, parameter_count


ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)
from mask_eee_rcnn.layers import cat


def kd_loss(mask_logit, refine_logit, mask_att, refine_att, instances, loss_type, temperature=0.1):

    if 'mask' in loss_type:
        eps = 1e-6
        cls_agnostic_mask = mask_logit.size(1) == 1
        total_num_masks = mask_logit.size(0)

        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
        
        gt_classes = []
        if not cls_agnostic_mask:
            for instances_per_image in instances:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)
        gt_classes = cat(gt_classes, dim=0)
        if cls_agnostic_mask:
            mask_logit = mask_logit[:, 0]
            refine_logit = refine_logit[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            # gt_classes = cat(gt_classes, dim=0)
            mask_logit = mask_logit[indices, gt_classes]
            refine_logit = refine_logit[indices, gt_classes]

        mask_logit = mask_logit.view(mask_logit.size(0), -1)
        refine_logit = refine_logit.view(refine_logit.size(0), -1)
        # here, the refine_logit is used as the target, so that gradient should not be backpropagated to refine_logit
        gt = refine_logit.detach()
        if loss_type == 'mask_kldiv':
            loss = F.kl_div(F.log_softmax(mask_logit / temperature, dim=1), F.softmax(gt / temperature, dim=1), reduction='batchmean') * (temperature ** 2)
        elif loss_type == 'mask_l1':
            loss = F.l1_loss(mask_logit, gt, reduction='mean')
        elif loss_type == 'mask_l2':
            loss = F.mse_loss(mask_logit, gt, reduction='mean')
        elif loss_type == 'mask_dice':
            gt = gt > 0.0
            mask_logit = mask_logit > 0.0
            # compute the dice loss
            intersection = (mask_logit * gt).sum()
            union = mask_logit.sum() + gt.sum()
            loss = 1 - (2 * intersection + eps) / (union + eps)
        else:
            raise NotImplementedError
        return loss
    elif 'sa' in loss_type:
        if mask_att.size(2) != refine_att.size(2):
            mask_att = F.interpolate(mask_att, size=refine_att.size()[2:], mode='bilinear', align_corners=False)

        mask_att = mask_att.view(mask_att.size(0), -1)
        refine_att = refine_att.view(refine_att.size(0), -1)
        if 'l1' in loss_type:
            loss = F.l1_loss(mask_att, refine_att.detach(), reduction='mean')
        elif 'l2' in loss_type:
            loss = F.mse_loss(mask_att, refine_att.detach(), reduction='mean')
        elif 'cosine' in loss_type:
            loss = 1 - F.cosine_similarity(mask_att, refine_att.detach(), dim=1).mean()
        else:
            raise NotImplementedError
        return loss





def dice_loss(pred1, pred2, instances, resolution=0.5):
    eps = 1e-6
    cls_agnostic_mask = pred1.size(1) == 1
    total_num_masks = pred1.size(0)

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
    
    gt_classes = []
    if not cls_agnostic_mask:
        for instances_per_image in instances:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
    gt_classes = cat(gt_classes, dim=0)
    if cls_agnostic_mask:
        pred1 = pred1[:, 0]
        pred2 = pred2[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        # gt_classes = cat(gt_classes, dim=0)
        pred1 = pred1[indices, gt_classes]
        pred2 = pred2[indices, gt_classes]
    if resolution != 1.0:
        pred1 = torch.functional.F.interpolate(pred1.unsqueeze(1), scale_factor=resolution, mode='bilinear', align_corners=False).squeeze(1)
        pred2 = torch.functional.F.interpolate(pred2.unsqueeze(1), scale_factor=resolution, mode='bilinear', align_corners=False).squeeze(1)

    pred2 = pred2 > 0.0
    pred1 = pred1 > 0.0
    # compute the dice loss
    intersection = (pred1 * pred2).sum()
    union = pred1.sum() + pred2.sum()
    loss = 1 - (2 * intersection + eps) / (union + eps)
    return loss


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
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
    for idx, proposals_per_image in enumerate(proposals):
        gt_classes = proposals_per_image.gt_classes
        try:
            proposal_boxes = proposals_per_image.proposal_boxes.tensor
        except AttributeError:
            proposal_boxes = proposals_per_image.pred_boxes.tensor
        proposal_width = proposal_boxes[:, 2] - proposal_boxes[:, 0]
        # proposal_height = proposal_boxes[:, 3] - proposal_boxes[:, 1]
        # print(proposal_width)
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label) & (proposal_width > 0)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
        
        eee_feat_cat = None
        eee_pred_cat = None
        box_refine_roi_cat = None
    
    return fg_proposals, fg_selection_masks, eee_feat_cat, eee_pred_cat, box_refine_roi_cat


def select_proposals_with_visible_keypoints(proposals):
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.
    Returns:
        proposals: only contains proposals with at least one visible keypoint.
    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.
    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection).squeeze(1)
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


def visualize_uncertainty(mask, uncertainty):
    if type(mask) is list:
        mask = mask[0]
    mask_np = mask.cpu().numpy().argmax(axis=1)
    uncertainty_np = uncertainty.sigmoid().cpu().numpy()[:,1,:,:]

    batch_size = mask_np.shape[0]
    for batch_idx in range(batch_size):
        mask_np_i = np.zeros_like(mask_np[batch_idx,:,:])
        uncertainty_np_i = np.zeros_like(uncertainty_np[batch_idx,:,:])
        mask_np_i[mask_np[batch_idx,:,:]==1] = 200
        uncertainty_np_i[uncertainty_np[batch_idx,:,:] > 0.5] = 200
        mask_np_cat = np.stack((mask_np_i, mask_np_i, mask_np_i), axis=0).transpose(1, 2, 0)
        uncertainty_np_cat = np.stack((uncertainty_np_i, uncertainty_np_i, uncertainty_np_i), axis=0).transpose(1, 2, 0)
        cv2.imwrite('vis_gt/{}.png'.format(batch_idx), np.hstack((mask_np_cat, uncertainty_np_cat)))




class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.
    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.
    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad() 
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        try:
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
        except AttributeError:
            print("label gt for visualization")
            pass

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            if proposals_per_image.has("proposal_boxes"):
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                sampled_idxs, gt_classes = self._sample_proposals(
                    matched_idxs, matched_labels, targets_per_image.gt_classes
                )

                # Set target attributes of the sampled proposals:
                proposals_per_image = proposals_per_image[sampled_idxs]
                proposals_per_image.gt_classes = gt_classes

                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                if has_gt:
                    sampled_targets = matched_idxs[sampled_idxs]
                    # NOTE: here the indexing waste some compute, because heads
                    # like masks, keypoints, etc, will filter the proposals again,
                    # (by foreground/background, or number of keypoints in the image, etc)
                    # so we essentially index the data twice.
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                            proposals_per_image.set(trg_name, trg_value[sampled_targets])
                else:
                    gt_boxes = Boxes(
                        targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                    )
                    proposals_per_image.gt_boxes = gt_boxes
            elif proposals_per_image.has("pred_boxes"):
                # Collect the predicted boxes and their attributes
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.pred_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                sampled_idxs, gt_classes = self._sample_proposals(
                    matched_idxs, matched_labels, targets_per_image.gt_classes
                )

                # Set target attributes of the sampled proposals:
                proposals_per_image = proposals_per_image[sampled_idxs]
                proposals_per_image.gt_classes = gt_classes

                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                if has_gt:
                    sampled_targets = matched_idxs[sampled_idxs]
                    # NOTE: here the indexing waste some compute, because heads
                    # like masks, keypoints, etc, will filter the proposals again,
                    # (by foreground/background, or number of keypoints in the image, etc)
                    # so we essentially index the data twice.
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                            proposals_per_image.set(trg_name, trg_value[sampled_targets])


            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        # storage = get_event_storage()
        # storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        # storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        # print(proposals_with_gt[0].get_fields())
        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        Returns:
            results (list[Instances]): length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.
            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = mask_rcnn_loss(mask_logits, proposals)
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            mask_logits = self.mask_head(x)
            mask_rcnn_inference(mask_logits, instances)
        return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.
    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self.input_shape = input_shape
        self._init_box_head(cfg)
        self._init_mask_head(cfg)
        self._init_maskiou_head(cfg)
        self._init_keypoint_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        self.box_eee_on = cfg.MODEL.ROI_BOX_HEAD.BOX_EEE_ON
        self.box_refine_on = cfg.MODEL.ROI_BOX_HEAD.BOX_REFINE_ON
        self.box_eee_error_type = cfg.MODEL.ROI_BOX_HEAD.BOX_EEE_ERROR_TYPE
        self.cls_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT
        self.box_loss_weight = cfg.MODEL.ROI_BOX_HEAD.BOX_LOSS_WEIGHT
        self.box_eee_loss_weight = cfg.MODEL.ROI_BOX_HEAD.BOX_EEE_LOSS_WEIGHT
        self.box_refine_loss_weight = cfg.MODEL.ROI_BOX_HEAD.BOX_REFINE_LOSS_WEIGHT
        self.class_refine_on = cfg.MODEL.ROI_BOX_HEAD.CLASS_REFINE_ON
        self.box_roi_feature_size = cfg.MODEL.ROI_BOX_HEAD.BOX_ROI_FEATURE_SIZE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        
        if self.box_eee_on or self.box_refine_on:
            self.box_eee_pooler = ROIPooler(
                output_size=pooler_resolution * 2, # 7x7 -> 14x14
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )


    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution), self.input_shape
        )


        self.mask_eee_on = cfg.MODEL.ROI_MASK_HEAD.MASK_EEE_ON
        self.mask_refine_on = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_ON
        self.mask_refine_num = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_NUM
        self.mask_refine_size = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_SIZE
        self.mask_refine_add_size = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_ADD_SIZE
        self.mask_loss_weight = cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_WEIGHT
        self.mask_eee_loss_weight = cfg.MODEL.ROI_MASK_HEAD.MASK_EEE_LOSS_WEIGHT
        self.mask_refine_loss_weight = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_LOSS_WEIGHT
        self.mask_refine_loss_change = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_LOSS_CHANGE
        self.mask_second_refine_loss_weight = cfg.MODEL.ROI_MASK_HEAD.MASK_SECOND_REFINE_LOSS_WEIGHT
        self.mask_eee_loss_type = cfg.MODEL.ROI_MASK_HEAD.MASK_EEE_LOSS_TYPE
        self.mask_eee_error_type = cfg.MODEL.ROI_MASK_HEAD.MASK_EEE_ERROR_TYPE
        self.monte_carlo_dropout_im = cfg.MODEL.ROI_MASK_HEAD.MONTE_CARLO_DROPOUT_IM
        self.monte_carlo_dropout_ee = cfg.MODEL.ROI_MASK_HEAD.MONTE_CARLO_DROPOUT_EE
        self.box_refine_on_mask_new_pooler = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_NEW_POOLER
        self.cls_logits_to_mask_feature_fusion = cfg.MODEL.ROI_BOX_HEAD.CLS_LOGITS_TO_MASK_FEATURE_FUSION
        self.cls_logits_to_box_fix_directly = cfg.MODEL.ROI_BOX_HEAD.CLS_LOGITS_TO_BOX_FIX_DIRECTLY
        self.cls_logits_to_mask_normalize_fusion = cfg.MODEL.ROI_BOX_HEAD.CLS_LOGITS_TO_MASK_NORMALIZE_FUSION
        self.mask_feat_size = cfg.MODEL.ROI_MASK_HEAD.MASK_FEAT_SIZE
        self.consistency_loss_on = cfg.MODEL.ROI_MASK_HEAD.CONSISTENCY_LOSS_ON
        self.consistency_loss_weight = cfg.MODEL.ROI_MASK_HEAD.CONSISTENCY_LOSS_WEIGHT
        self.consistency_loss_resolution = cfg.MODEL.ROI_MASK_HEAD.CONSISTENCY_LOSS_RESOLUTION
        self.box_refine_on_mask_head = cfg.MODEL.ROI_MASK_HEAD.BOX_REFINE_ON_MASK_HEAD
        self.kd_loss_on = cfg.MODEL.ROI_MASK_HEAD.KD_LOSS_ON
        self.kd_loss_weight = cfg.MODEL.ROI_MASK_HEAD.KD_LOSS_WEIGHT
        self.kd_loss_temperature = cfg.MODEL.ROI_MASK_HEAD.KD_LOSS_TEMPERATURE
        self.kd_loss_type = cfg.MODEL.ROI_MASK_HEAD.KD_LOSS_TYPE
        self.kd_loss_warmup_iter = cfg.MODEL.ROI_MASK_HEAD.KD_LOSS_WARMUP_ITER
        self.sem_seg_on = cfg.MODEL.SEM_SEG_HEAD.SEM_SEG_ON
        self.sem_seg_to_dualfix = cfg.MODEL.ROI_MASK_HEAD.SEM_SEG_TO_DUALFIX
        self.error_estimation_class_agnostic = cfg.MODEL.ROI_MASK_HEAD.ERROR_ESTIMATION_CLASS_AGNOSTIC
        self.mask_dct_on = cfg.MODEL.ROI_MASK_HEAD.MASK_DCT_ON
        self.mask_dct_after_maskfix_on = cfg.MODEL.ROI_MASK_HEAD.MASK_DCT_AFTER_MASKFIX_ON
        self.deconv_before_maskfix = cfg.MODEL.ROI_MASK_HEAD.DECONV_BEFORE_MASKFIX
        self.i_fpn_on = cfg.MODEL.ROI_MASK_HEAD.I_FPN_ON
        self.combine_dice_and_ce_loss = cfg.MODEL.ROI_MASK_HEAD.COMBINE_DICE_AND_CE_LOSS
        self.mask_loss_type = cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_TYPE
        self.mask_dct_loss_type = cfg.MODEL.ROI_MASK_HEAD.MASK_DCT_LOSS_TYPE
        self.boundary_preserving_on = cfg.MODEL.ROI_MASK_HEAD.BOUNDARY_PRESERVING_ON
        self.refinemask_on = cfg.MODEL.ROI_MASK_HEAD.REFINEMASK_ON
        self.boundary_error_on = cfg.MODEL.ROI_MASK_HEAD.BOUNDARY_ERROR_ON
        self.mask_scoring_at_error_estimation = cfg.MODEL.ROI_MASK_HEAD.MASK_SCORING_AT_ERROR_ESTIMATION
        self.maskiou_weight = cfg.MODEL.MASKIOU_LOSS_WEIGHT
        self.patchdct_on = cfg.MODEL.ROI_MASK_HEAD.PATCHDCT_ON
        self.iter = 0

        if len(cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_WEIGHT_LIST) > 0:
            self.mask_loss_weight_list = cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_WEIGHT_LIST
        if len(cfg.MODEL.ROI_MASK_HEAD.MASK_EEE_LOSS_WEIGHT_LIST) > 0:
            self.mask_eee_loss_weight_list = cfg.MODEL.ROI_MASK_HEAD.MASK_EEE_LOSS_WEIGHT_LIST 

        if not self.mask_refine_loss_change:
            self.mask_second_refine_loss_weight = self.mask_refine_loss_weight
        

        if (self.mask_eee_on or self.mask_refine_on) and (self.mask_refine_size != 14):
            self.use_roi_pooler_l = True
            if self.i_fpn_on:
                self.mask_eee_pooler = ROIPooler(
                    # output_size=pooler_resolution * 2, # 14x14 -> 28x28
                    output_size=self.mask_refine_size,
                    scales=(pooler_scales[1],),
                    sampling_ratio=sampling_ratio,
                    pooler_type=pooler_type,
                )
            else:
                self.mask_eee_pooler = ROIPooler(
                    output_size=pooler_resolution * 2, # 14x14 -> 28x28
                    # output_size=self.mask_refine_size,
                    scales=pooler_scales,
                    sampling_ratio=sampling_ratio,
                    pooler_type=pooler_type,
                )
        else:
            self.use_roi_pooler_l = False
        
        if self.boundary_preserving_on and not self.mask_refine_on:
            # boundary_resolution     = cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION
            # boundary_in_features    = cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES
            # POOLER_RESOLUTION: 28
            # IN_FEATURES: ["p2"]
            # self.feature_strides = {k: v.stride for k, v in input_shape_pooler.items()}
            boundary_resolution = 28
            boundary_in_features = ["p2"]
            self.boundary_in_features = boundary_in_features
            boundary_scales = tuple(1.0 / self.feature_strides[k] for k in boundary_in_features)
            sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
            pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

            self.boundary_pooler = ROIPooler(
                output_size=boundary_resolution,
                scales=boundary_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type
            )

            fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
            self.box_predictor = FastRCNNOutputLayers(
                fc_dim, self.num_classes, self.cls_agnostic_bbox_reg
            )


        if self.deconv_before_maskfix:
            # self.use_roi_pooler_l = True
            if self.i_fpn_on:
                self.mask_eee_pooler_xl = ROIPooler(
                    # output_size=pooler_resolution * 2, # 14x14 -> 28x28
                    output_size=self.mask_refine_size * 2,  # 28x28 -> 56x56
                    scales=(pooler_scales[0],),
                    sampling_ratio=sampling_ratio,
                    pooler_type=pooler_type,
                )
            else:
                self.mask_eee_pooler_xl = ROIPooler(
                    # output_size=pooler_resolution * 2, # 14x14 -> 28x28
                    output_size=self.mask_refine_size * 2,  # 28x28 -> 56x56
                    scales=pooler_scales,
                    sampling_ratio=sampling_ratio,
                    pooler_type=pooler_type,
                )
            self.use_roi_pooler_xl = True
        else:
            self.use_roi_pooler_xl = False
        
        if self.sem_seg_on and self.sem_seg_to_dualfix:
            
            # import time
            # time.sleep(1000)
            self.mask_sem_seg_pooler_s = ROIPooler(
                output_size=pooler_resolution,
                scales=(1.0,),
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            self.mask_sem_seg_pooler_l = ROIPooler(
                output_size=pooler_resolution * 2, # 14x14 -> 28x28
                scales=(1.0,),
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )


    def _init_maskiou_head(self, cfg):
        self.maskiou_on = cfg.MODEL.MASKIOU_ON
        if not self.maskiou_on:
            return

        self.maskiou_head = build_maskiou_head(cfg)
        self.maskiou_weight = cfg.MODEL.MASKIOU_LOSS_WEIGHT


    def _init_keypoint_head(self, cfg):
        # fmt: off
        self.keypoint_on                         = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution                        = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales                            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # noqa
        sampling_ratio                           = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type                              = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )


    def forward(self, images, features, proposals, targets=None, sem_pred=None, sem_features=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets
        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses, feat_b2m, class_logits = self._forward_box(features_list, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            if self.maskiou_on and self.mask_eee_on:
                loss, mask_features, selected_mask, labels, maskiou_targets = self._forward_mask(features_list, proposals)
                losses.update(loss)
                losses.update(self._forward_maskiou(mask_features, proposals, selected_mask, labels, maskiou_targets))
            elif self.maskiou_on:
                loss, mask_features, selected_mask, labels, maskiou_targets = self._forward_mask(features_list, proposals)
                losses.update(loss)
                losses.update(self._forward_maskiou(mask_features, proposals, selected_mask, labels, maskiou_targets))
            elif self.mask_eee_on:
                loss = self._forward_mask(features_list, proposals, targets=targets)
                losses.update(loss)
            else:
                losses.update(self._forward_mask(features_list, proposals, targets=targets))
            
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        else:
            pred_instances, feat_b2m, class_logits = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.
        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        features = [features[f] for f in self.in_features]

        if self.maskiou_on and self.mask_eee_on:
            instances, mask_features = self._forward_mask(features, instances)
            instances = self._forward_maskiou(mask_features, instances)
        elif self.maskiou_on:
            instances, mask_features = self._forward_mask(features, instances)
            instances = self._forward_maskiou(mask_features, instances)
        elif self.mask_eee_on:
            instances = self._forward_mask(features, instances)
        else:
            instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances


    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        if self.box_eee_on or self.box_refine_on:
            if self.box_roi_feature_size == "l":
                box_features_l = self.box_eee_pooler(features, [x.proposal_boxes for x in proposals])
            else:
                box_features_l = None
        else:
            box_features_l = None
        pred_class_logits, pred_proposal_deltas, eee_proposal_deltas, refine_class_logits, refine_proposal_deltas, feat_b2m  = self.box_head(box_features, box_features_l)

        if self.boundary_preserving_on and not self.mask_refine_on:
            pred_class_logits, pred_proposal_deltas = self.box_predictor(pred_class_logits)

        del box_features
        del box_features_l

        if self.training:
            outputs = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                loss_weight=self.box_loss_weight,
                cls_loss_weight=self.cls_loss_weight
            )


            if self.train_on_pred_boxes:
                pred_boxes = outputs.predict_boxes_for_gt_classes()
                for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                    proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
                losses = outputs.losses()
                
            else:
                losses = outputs.losses()
                
            
            return losses, feat_b2m, pred_class_logits
        
        else:                
            outputs = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                loss_weight=self.box_loss_weight,
                feat_b2m=feat_b2m
            )
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            
            # for idx in range(len(pred_instances)):
            #     for ten_idx in range(pred_instances[idx].pred_boxes_initial.tensor.shape[0]):
            #         print('initial box: ', pred_instances[idx].pred_boxes_initial.tensor.cpu().numpy()[ten_idx])
            #         print('box: ', pred_instances[idx].pred_boxes.tensor.cpu().numpy()[ten_idx])

            if feat_b2m is not None:
                for key in feat_b2m.keys():
                    if key == 'conv':
                        feat_b2m[key] = torch.cat([p.feat_b2m_conv for p in pred_instances], dim=0)
                    elif key == 'fc':
                        feat_b2m[key] = torch.cat([p.feat_b2m_fc for p in pred_instances], dim=0)
            if self.cls_logits_to_mask_feature_fusion or self.cls_logits_to_box_fix_directly or self.cls_logits_to_mask_normalize_fusion:
                class_logits = torch.cat([p.class_logits for p in pred_instances], dim=0)
            else:
                class_logits = None

            return pred_instances, feat_b2m, class_logits


    def _forward_mask(self, features, instances, images=None, targets=None):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _, _, _, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features_s = self.mask_pooler(features, proposal_boxes)
            if self.i_fpn_on:
                features_l = self.mask_eee_pooler([features[1]], proposal_boxes) if self.use_roi_pooler_l else None
                features_xl = self.mask_eee_pooler_xl([features[0]], proposal_boxes) if self.use_roi_pooler_xl else None
            else:
                features_l = self.mask_eee_pooler(features, proposal_boxes) if self.use_roi_pooler_l else None
                features_xl = self.mask_eee_pooler_xl(features, proposal_boxes) if self.use_roi_pooler_xl else None
            
            if self.boundary_preserving_on and not self.mask_refine_on:
                features_l = self.boundary_pooler([features[0]], proposal_boxes)

            if self.mask_eee_on:
                # !TODO: support maskiou_on with mask_eee_on
                losses = {}
                mask_logits_list, eee_logits_list, mask_refine_logits, box_refine_list, pred_instances_list, boundary_logits, patchdct_logits, proposals_list, _  = self.mask_head(
                    features_s, features_l, features_xl, proposals=proposals, features=features, targets=targets)
                
                if self.mask_dct_on:
                    losses['loss_mask'] = mask_rcnn_dct_loss(mask_logits[0], proposals, self.mask_head.dct_encoding, mask_loss_para=self.mask_loss_weight)
                else:
                    for idx, mask_logits in enumerate(mask_logits_list):
                        if self.mask_scoring_at_error_estimation:
                            loss_mask, selected_mask, labels, maskiou_targets = mask_rcnn_loss(mask_logits, proposals, self.mask_scoring_at_error_estimation, mask_loss_type=self.mask_loss_type, combine_dice_ce=self.combine_dice_and_ce_loss)
                        else:
                            if idx == 0:
                                loss_mask = mask_rcnn_loss(mask_logits, proposals, self.mask_scoring_at_error_estimation, mask_loss_type=self.mask_loss_type, combine_dice_ce=self.combine_dice_and_ce_loss)
                            else:
                                loss_mask = mask_rcnn_loss(mask_logits, proposals_list[idx-1], self.mask_scoring_at_error_estimation, 
                                                           pred_instances=pred_instances_list[idx-1], mask_loss_type=self.mask_loss_type, combine_dice_ce=self.combine_dice_and_ce_loss)
                        
                        if type(self.mask_loss_weight) == list:
                            losses['loss_mask_{}'.format(idx)] = loss_mask * self.mask_loss_weight[idx]
                        else:
                            losses['loss_mask_{}'.format(idx)] = loss_mask * self.mask_loss_weight
                
                if self.box_refine_on_mask_head:
                    for idx, box_refine in enumerate(box_refine_list):
                        refine_box_losses = box_refine.losses()
                        for k, v in refine_box_losses.items():
                            losses[k + '_{}'.format(idx)] = v
                        
                refine_idx = 0
                if self.mask_dct_on:
                    losses['loss_eee_{}'.format(refine_idx)] = mask_rcnn_eee_loss(mask_logits[1], eee_logits, proposals, self.mask_eee_loss_type, 
                                                                                    self.mask_eee_error_type, combine_dice_ce=self.combine_dice_and_ce_loss) * self.mask_eee_loss_weight
                else:
                    for eee_idx, eee_logits in enumerate(eee_logits_list):
                        if type(self.mask_eee_loss_weight) == list:
                            mask_eee_loss_weight = self.mask_eee_loss_weight[eee_idx]
                        else:
                            mask_eee_loss_weight = self.mask_eee_loss_weight
                        mask_logits = mask_logits_list[eee_idx]
                        if eee_idx == 0:
                            losses['loss_eee_{}'.format(eee_idx)] = mask_rcnn_eee_loss(mask_logits, eee_logits, proposals, self.mask_eee_loss_type, 
                                                                                        self.mask_eee_error_type, boundary_error_on=self.boundary_error_on) * mask_eee_loss_weight
                        else:
                            losses['loss_eee_{}'.format(eee_idx)] = mask_rcnn_eee_loss(mask_logits, eee_logits, proposals_list[eee_idx-1], self.mask_eee_loss_type, 
                                                                                        self.mask_eee_error_type, boundary_error_on=self.boundary_error_on, pred_instances=pred_instances_list[eee_idx-1]) * mask_eee_loss_weight
                
                if self.mask_refine_on:
                    if len(pred_instances_list) != 0:
                        pred_instances = pred_instances_list[-1]
                    else:
                        pred_instances = None
                    if len(proposals_list) != 0:
                        proposals = proposals_list[-1]
                    # else:
                    #     proposals = None
                    
                    if self.mask_dct_after_maskfix_on:
                        loss_refine = mask_rcnn_dct_loss(mask_refine_logits, proposals, self.mask_head.dct_encoding, pred_instances=pred_instances)
                    elif self.boundary_preserving_on:
                        loss_refine_mask, loss_refine_boundary = boundary_preserving_mask_loss(mask_refine_logits, boundary_logits, proposals, pred_instances=pred_instances)
                        loss_refine = loss_refine_mask + loss_refine_boundary
                    elif self.refinemask_on:
                        stage_mask_targets = self.mask_head.refinemask_get_targets(mask_refine_logits, instances=proposals, pred_instances=pred_instances)
                        loss_refine = self.mask_head.refinemask_loss_func(mask_refine_logits, stage_mask_targets)
                    elif self.patchdct_on:
                        bfg, patch_vectors = patchdct_logits
                        loss_refine = self.mask_head.patchdct_mask_rcnn_dct_loss(mask_refine_logits, bfg, patch_vectors, instances=proposals, pred_instances=pred_instances)
                    else:
                        if self.maskiou_on:
                            loss_refine, selected_mask, labels, maskiou_targets = mask_rcnn_loss(mask_refine_logits, proposals, self.maskiou_on, name="mask_refine_loss", 
                                                            pred_instances=pred_instances, images=images, mask_loss_type=self.mask_loss_type, combine_dice_ce=self.combine_dice_and_ce_loss)  
                        else:
                            loss_refine = mask_rcnn_loss(mask_refine_logits, proposals, self.maskiou_on, name="mask_refine_loss", 
                                                            pred_instances=pred_instances, images=images, mask_loss_type=self.mask_loss_type, combine_dice_ce=self.combine_dice_and_ce_loss)                          
                    losses['loss_refine_mask_{}'.format(refine_idx)] = loss_refine * self.mask_refine_loss_weight

                if self.maskiou_on:
                    return losses, features_s, selected_mask, labels, maskiou_targets
                else:
                    return losses
            elif self.mask_refine_on:
                losses = {}
                mask_logits_list, eee_logits_list, mask_refine_logits, box_refine_list, pred_instances_list, boundary_logits, patchdct_logits, proposals_list  = self.mask_head(
                    features_s, features_l, features_xl, proposals=proposals, features=features, targets=targets)
                
                for idx, mask_logits in enumerate(mask_logits_list):
                    if idx == 0:
                        loss_mask = mask_rcnn_loss(mask_logits, proposals, self.mask_scoring_at_error_estimation, mask_loss_type=self.mask_loss_type, combine_dice_ce=self.combine_dice_and_ce_loss)
                    else:
                        loss_mask = mask_rcnn_loss(mask_logits, proposals_list[idx-1], self.mask_scoring_at_error_estimation, 
                                                    pred_instances=pred_instances_list[idx-1], mask_loss_type=self.mask_loss_type, combine_dice_ce=self.combine_dice_and_ce_loss)
                
                    if type(self.mask_loss_weight) == list:
                        losses['loss_mask_{}'.format(idx)] = loss_mask * self.mask_loss_weight[idx]
                    else:
                        losses['loss_mask_{}'.format(idx)] = loss_mask * self.mask_loss_weight
                
                # losses['loss_mask'] = mask_rcnn_loss(mask_logits, proposals, self.maskiou_on) * self.mask_loss_weight
                # losses['loss_eee'] = mask_rcnn_eee_loss(mask_logits, eee_logits, proposals, self.mask_eee_loss_type, self.mask_eee_error_type) * self.mask_eee_loss_weight
                if self.box_refine_on_mask_head:
                    for idx, box_refine in enumerate(box_refine_list):
                        refine_box_losses = box_refine.losses()
                        for k, v in refine_box_losses.items():
                            losses[k + '_{}'.format(idx)] = v
                if self.mask_refine_on:
                    refine_idx = 0
                    if len(pred_instances_list) != 0:
                        pred_instances = pred_instances_list[-1]
                    else:
                        pred_instances = None
                    if len(proposals_list) != 0:
                        proposals = proposals_list[-1]
                    # if self.box_refine_on_mask_head:
                    #     pred_instances = pred_instances_list[-1]
                    # else:
                    #     pred_instances = None
                    losses['loss_refine_mask_{}'.format(refine_idx)] = mask_rcnn_loss(mask_refine_logits, proposals, self.maskiou_on, pred_instances=pred_instances, name="mask_refine_loss") * self.mask_refine_loss_weight
                return losses
                # mask_logits, _, _, _, _, _ = self.mask_head(features_s, features_l)
                # return {"loss_mask": mask_rcnn_loss(mask_logits, proposals, self.maskiou_on)}
            elif self.maskiou_on:
                mask_logits, _, _, _, _, _, _, _ = self.mask_head(features_s)
                loss_mask, selected_mask, labels, maskiou_targets = mask_rcnn_loss(mask_logits, proposals, self.maskiou_on)
                return {"loss_mask": loss_mask}, features_s, selected_mask, labels, maskiou_targets
            else:
                mask_logits_list, _, _, _, _, _, patchdct_logits, _, _ = self.mask_head(features_s, features_l, proposals=proposals, features=features)
                mask_logits = mask_logits_list[0]
                if self.mask_dct_on:
                    return {"loss_mask": mask_rcnn_dct_loss(mask_logits, proposals, self.mask_head.dct_encoding, dct_loss_type=self.mask_dct_loss_type, mask_loss_para=self.mask_loss_weight)}
                elif self.refinemask_on:
                    stage_mask_targets = self.mask_head.refinemask_get_targets(mask_logits, instances=proposals)
                    loss_mask = self.mask_head.refinemask_loss_func(mask_logits, stage_mask_targets)
                    return {"loss_mask": loss_mask}
                elif self.patchdct_on:
                    bfg, patch_vectors = patchdct_logits
                    loss_mask = self.mask_head.patchdct_mask_rcnn_dct_loss(mask_logits, bfg, patch_vectors, instances=proposals)
                    return {"loss_mask": loss_mask}
                elif self.boundary_preserving_on:
                    loss_mask, loss_boundary = boundary_preserving_mask_loss(
                        mask_logits, boundary_logits, instances=proposals)
                    return {"loss_mask": loss_mask,
                            "loss_boundary": loss_boundary}
                return {"loss_mask": mask_rcnn_loss(mask_logits, proposals, self.maskiou_on)}

        else:
            pred_boxes = [x.pred_boxes for x in instances]
            features_s = self.mask_pooler(features, pred_boxes)
            if self.i_fpn_on:
                features_l = self.mask_eee_pooler([features[1]], pred_boxes) if self.use_roi_pooler_l else None
                features_xl = self.mask_eee_pooler_xl([features[0]], pred_boxes) if self.use_roi_pooler_xl else None
            else:
                features_l = self.mask_eee_pooler(features, pred_boxes) if self.use_roi_pooler_l else None
                features_xl = self.mask_eee_pooler_xl(features, pred_boxes) if self.use_roi_pooler_xl else None
            
            if self.boundary_preserving_on and not self.mask_refine_on:
                features_l = self.boundary_pooler([features[0]], pred_boxes)

            mask_logits_list, eee_logits_list, mask_refine_logits, box_refine_list, pred_instances_list, boundary_logits, patchdct_logits, proposals_list, pca_list = self.mask_head(
                features_s, features_l, features_xl, proposals=instances, features=features)

            mask_logits = mask_logits_list[-1]

            if mask_refine_logits is not None:
                mask_logits = mask_logits_list[-1]
                eee_logits = eee_logits_list[-1] if self.mask_eee_on else None
                pred_instances = pred_instances_list[-1] if self.box_refine_on_mask_head else None

                if self.mask_dct_after_maskfix_on:
                    mask_refine_logits = self.mask_head.mask_rcnn_dct_inference(mask_refine_logits, pred_instances, instances)
                elif self.refinemask_on:
                    mask_refine_logits = self.mask_head.refinemask_get_seg_masks(mask_refine_logits, pred_instances, instances)
                elif self.patchdct_on:
                    bfg, patch_vectors = patchdct_logits
                    mask_refine_logits = self.mask_head.patchdct_mask_rcnn_dct_inference(mask_refine_logits, bfg, patch_vectors, pred_instances, instances)
                mask_rcnn_inference(mask_refine_logits, instances, mask_logits, eee_logits, refined_box_mask_head=pred_instances, mask_dct_on=self.mask_dct_after_maskfix_on, patch_dct_on=self.patchdct_on, feat_pca=pca_list)
            else:
                # pred_instances = pred_instances_list[0]
                pred_instances = None
                if self.mask_dct_on:
                    mask_logits = self.mask_head.mask_rcnn_dct_inference(mask_logits, pred_instances, instances)
                elif self.refinemask_on:
                    mask_logits = self.mask_head.refinemask_get_seg_masks(mask_logits, pred_instances, instances)
                elif self.patchdct_on:
                    bfg, patch_vectors = patchdct_logits
                    mask_logits = self.mask_head.patchdct_mask_rcnn_dct_inference(mask_logits, bfg, patch_vectors, pred_instances, instances)
                mask_rcnn_inference(mask_logits, instances, mask_dct_on=self.mask_dct_on, patch_dct_on=self.patchdct_on, feat_pca=pca_list)
            if self.maskiou_on:
                return instances, features_s
            else:
                return instances


    def _forward_maskiou(self, mask_features, instances, selected_mask=None, labels=None, maskiou_targets=None):
        """
        Forward logic of the mask iou prediction branch.
        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, calibrate instances' scores.
        """
        if not self.maskiou_on:
            return {} if self.training else instances

        if self.training:
            pred_maskiou = self.maskiou_head(mask_features, selected_mask)
            return {"loss_maskiou": mask_iou_loss(labels, pred_maskiou, maskiou_targets, self.maskiou_weight)}

        else:
            masks = torch.cat([i.pred_masks for i in instances], 0)
            if masks.shape[0] == 0:
                return instances
            pred_maskiou = self.maskiou_head(mask_features, masks)
            mask_iou_inference(instances, pred_maskiou)
            return instances


    def _forward_keypoint(self, features, instances):
        """
        Forward logic of the keypoint prediction branch.
        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)

            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances