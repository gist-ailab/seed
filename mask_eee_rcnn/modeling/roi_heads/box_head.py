# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, FastRCNNEELayers
from mask_eee_rcnn.layers import Conv2d, ShapeSpec, get_norm
from mask_eee_rcnn.utils.registry import Registry

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.box_eee_on = cfg.MODEL.ROI_BOX_HEAD.BOX_EEE_ON
        self.box_refine_on = cfg.MODEL.ROI_BOX_HEAD.BOX_REFINE_ON
        self.box_eee_error_type = cfg.MODEL.ROI_BOX_HEAD.BOX_EEE_ERROR_TYPE
        self.box_loss_weight = cfg.MODEL.ROI_BOX_HEAD.BOX_LOSS_WEIGHT
        self.box_eee_loss_weight = cfg.MODEL.ROI_BOX_HEAD.BOX_EEE_LOSS_WEIGHT
        self.box_refine_loss_weight = cfg.MODEL.ROI_BOX_HEAD.BOX_REFINE_LOSS_WEIGHT
        self.class_refine_on = cfg.MODEL.ROI_BOX_HEAD.CLASS_REFINE_ON
        self.box_dense_fusion = cfg.MODEL.ROI_BOX_HEAD.BOX_DENSE_FUSION
        self.box_roi_feature_size = cfg.MODEL.ROI_BOX_HEAD.BOX_ROI_FEATURE_SIZE
        self.box_to_mask_feature_fusion = cfg.MODEL.ROI_BOX_HEAD.BOX_TO_MASK_FEATURE_FUSION
        self.box_to_boxfix_feature_fusion = cfg.MODEL.ROI_BOX_HEAD.BOX_TO_BOXFIX_FEATURE_FUSION
        self.box_to_boxfix_fc_feature_fusion = cfg.MODEL.ROI_BOX_HEAD.BOX_TO_BOXFIX_FC_FEATURE_FUSION
        
        self.mask_refine_on = cfg.MODEL.ROI_MASK_HEAD.MASK_REFINE_ON
        self.boundary_preserving_on = cfg.MODEL.ROI_MASK_HEAD.BOUNDARY_PRESERVING_ON
        
        num_fusion_fc = 1
        error_dim = 2
        box_dim = 4
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        if self.box_roi_feature_size == "l":
            self._output_size_eee = (input_shape.channels, input_shape.height*2, input_shape.width*2)
            self._output_size_refine = (input_shape.channels, input_shape.height*2, input_shape.width*2)
        else:
            self._output_size_eee = (input_shape.channels, input_shape.height, input_shape.width)
            self._output_size_refine = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        if not self.boundary_preserving_on or self.mask_refine_on:
            self.box_predictor = FastRCNNOutputLayers(
                self._output_size, self.num_classes, self.cls_agnostic_bbox_reg
            )

        if self.box_eee_on or self.box_refine_on:
            self.conv_norm_relus_refine = []
            for k in range(num_conv):
                conv = Conv2d(
                    self._output_size_refine[0],
                    conv_dim,
                    kernel_size=3,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dim),
                    activation=F.relu,
                )
                self.add_module("conv_refine{}".format(k + 1), conv)
                self.conv_norm_relus_refine.append(conv)
                self._output_size_refine = (conv_dim, self._output_size_refine[1], self._output_size_refine[2])
                self._output_size_eee = (conv_dim, self._output_size_eee[1], self._output_size_eee[2])
            for layer in self.conv_norm_relus_refine:
                weight_init.c2_msra_fill(layer)

        if self.box_eee_on:
            self.fcs_i2e_fusion = []
            for k in range(num_fusion_fc):
                fusion_size = np.prod(self._output_size_eee) + fc_dim + self.num_classes * box_dim
                fc = nn.Linear(fusion_size, fc_dim)
                self.add_module("fc_i2e_fusion{}".format(k + 1), fc)
                self.fcs_i2e_fusion.append(fc)
                self._output_size_eee = fc_dim
            
            self.fcs_eee = []
            for k in range(num_fc):
                fc = nn.Linear(np.prod(self._output_size_eee), fc_dim)
                self.add_module("fc_eee{}".format(k + 1), fc)
                self.fcs_eee.append(fc)
                self._output_size_eee = fc_dim
            
            for layer in self.fcs_i2e_fusion:
                weight_init.c2_xavier_fill(layer)
            for layer in self.fcs_eee:
                weight_init.c2_xavier_fill(layer)
            
            self.box_eee_predictor = FastRCNNEELayers(
                self._output_size_eee, error_dim, self.cls_agnostic_bbox_reg, error_type=self.box_eee_error_type
            )
        
        if self.box_refine_on:
            self.fcs_e2r_fusion = []
            for k in range(num_fusion_fc):
                if self.box_dense_fusion:
                    if self.box_eee_error_type == 'score':
                        fusion_size = np.prod(self._output_size_refine) + 2 * fc_dim + self.num_classes * box_dim + 1
                    else:
                        fusion_size = np.prod(self._output_size_refine) + 2 * fc_dim + self.num_classes * box_dim + error_dim * box_dim
                else:
                    fusion_size = np.prod(self._output_size_refine) + fc_dim + error_dim * box_dim
                fc = nn.Linear(fusion_size, fc_dim)
                self.add_module("fc_e2r_fusion{}".format(k + 1), fc)
                self.fcs_e2r_fusion.append(fc)
                self._output_size_refine = fc_dim

            self.fcs_refine = []
            for k in range(num_fc):
                fc = nn.Linear(fc_dim, fc_dim)
                self.add_module("fc_refine{}".format(k + 1), fc)
                self.fcs_refine.append(fc)
                self._output_size_refine = fc_dim
            
            for layer in self.fcs_e2r_fusion:
                weight_init.c2_msra_fill(layer)
            for layer in self.fcs_refine:
                weight_init.c2_xavier_fill(layer)
            
            self.box_refine_predictor = FastRCNNOutputLayers(
                self._output_size_refine, self.num_classes, self.cls_agnostic_bbox_reg, class_refine=self.class_refine_on
            )


    def forward(self, x_s, x_l=None):
        """
        x_s: 7x7
        x_l: 14x14
        """
        feat_b2m = None
        for idx, layer in enumerate(self.conv_norm_relus):
            if idx == 0:
                x = layer(x_s)
                if self.box_to_mask_feature_fusion or self.box_to_boxfix_feature_fusion:
                    feat_b2m = {}
                    feat_b2m["conv"] = x.clone()
            else:
                x = layer(x)
        if len(self.fcs):
            try:
                if x.dim() > 2:
                    x = torch.flatten(x, start_dim=1)
            except UnboundLocalError:
                x = torch.flatten(x_s, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
            if self.box_to_boxfix_fc_feature_fusion:
                try:
                    feat_b2m["fc"] = x.clone()
                except TypeError:
                    feat_b2m = {}
                    feat_b2m["fc"] = x.clone()
            if not self.boundary_preserving_on or self.mask_refine_on:
                pred_class_logits, pred_proposal_deltas = self.box_predictor(x)
            else:
                pred_class_logits = x
                pred_proposal_deltas = None

        if self.box_eee_on or self.box_refine_on:
            if self.box_roi_feature_size == "l":
                for idx, layer in enumerate(self.conv_norm_relus_refine):
                    if idx == 0:
                        x_re = layer(x_l)
                    else:
                        x_re = layer(x_re)
                try:
                    if x_re.dim() > 2:
                        x_re = torch.flatten(x_re, start_dim=1)
                except UnboundLocalError:
                    x_re = torch.flatten(x_l, start_dim=1)
            else:
                for idx, layer in enumerate(self.conv_norm_relus_refine):
                    if idx == 0:
                        x_re = layer(x_s)
                    else:
                        x_re = layer(x_re)
                try:
                    if x_re.dim() > 2:
                        x_re = torch.flatten(x_re, start_dim=1)
                except UnboundLocalError:
                    x_re = torch.flatten(x_s, start_dim=1)
            
        if self.box_eee_on:
            x_eee = torch.cat((x_re, x, pred_proposal_deltas), dim=1)
            for layer in self.fcs_i2e_fusion:
                x_eee = F.relu(layer(x_eee))
            for layer in self.fcs_eee:
                x_eee = F.relu(layer(x_eee))
            eee_proposal_deltas = self.box_eee_predictor(x_eee)
        else:
            eee_proposal_deltas = None

        if self.box_refine_on:
            if self.box_dense_fusion:
                x_refine = torch.cat((x_re, x, pred_proposal_deltas, x_eee, eee_proposal_deltas), dim=1)
            else:
                x_refine = torch.cat((x_re, x_eee, eee_proposal_deltas), dim=1)
            for layer in self.fcs_e2r_fusion:
                x_refine = F.relu(layer(x_refine))
            for layer in self.fcs_refine:
                x_refine = F.relu(layer(x_refine))
            refine_class_logits, refine_proposal_deltas = self.box_refine_predictor(x_refine)
        else:
            refine_class_logits = None
            refine_proposal_deltas = None

        return pred_class_logits, pred_proposal_deltas, eee_proposal_deltas, refine_class_logits, refine_proposal_deltas, feat_b2m

    @property
    def output_size(self):
        if self.box_eee_on or self.box_refine_on:
            return self._output_size_refine
        else:
            return self._output_size


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
