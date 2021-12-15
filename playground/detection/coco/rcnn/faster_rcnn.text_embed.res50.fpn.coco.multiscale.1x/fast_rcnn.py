# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from cvpods.layers import cat, generalized_batched_nms
from cvpods.modeling.losses import smooth_l1_loss
from cvpods.structures import Boxes, Instances
from cvpods.utils import get_event_storage

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.class_proj = nn.Linear(input_size, 512)
        self.class_embed = nn.Parameter(torch.empty((num_classes + 1, 512)))
        self.text_embed = nn.Parameter(torch.empty((num_classes + 1, 512)), requires_grad=False)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.class_embed, std=0.01)
        with torch.no_grad():
            self.text_embed.data = torch.load("text_embedding.pth", map_location='cpu')
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        cls_feature = self.class_proj(x)
        cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
        cls_embed = self.class_embed / self.class_embed.norm(dim=-1, keepdim=True)
        cls_embed = 0.5 * cls_embed + 0.5 * self.text_embed
        logit_scale = self.logit_scale.exp()
        scores = logit_scale * cls_feature @ cls_embed.t()

        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas
