# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer

from ...builder import ROTATED_HEADS
from .rotated_bbox_head import RotatedBBoxHead
from .metrics import ArcMarginProduct
from .focal_loss import  FocalLoss
import math
import torch.nn.functional as F
from .partial_fc_v2 import PartialFC_V2,PartialFC_V3
from .CombinedMarginLoss import CombinedMarginLoss
import os
from torch import distributed
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from torch.nn.functional import linear, normalize

@ROTATED_HEADS.register_module()
class RotatedConvFCBBoxHead(RotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg

    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(RotatedConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)

        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


def increase_classification_boundaries( logits, labels,hard_thr):
    index_positive = torch.where(labels != -1)[0]
    target_logit = logits[index_positive, labels[index_positive].view(-1)]
    final_target_logit = target_logit - hard_thr
    logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
    return logits

@ROTATED_HEADS.register_module()
class RotatedConvFCBBoxHead_hard(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024,
                hard_thr = 0.2,
                *args, **kwargs):
        super(RotatedConvFCBBoxHead_hard, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        # assert self.num_classes == self.num_classes1 + self.num_classes2
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
                # cls_channels2 = self.loss_cls.get_cls_channels(self.num_classes2)# TODO insert
            else:
                cls_channels = self.num_classes + 1
                # cls_channels2 = self.num_classes2 + 1  #TODO insert
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)

        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)
        self.hard_thr = hard_thr

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                cls_score = increase_classification_boundaries( cls_score, labels,self.hard_thr)
                loss_cls1_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                
                if isinstance(loss_cls1_, dict):
                    losses.update(loss_cls1_)
                else:
                    losses['loss_cls'] = loss_cls1_
      
                if self.custom_activation:
                    acc1_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc1_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)

                
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
    

@ROTATED_HEADS.register_module()
class RotatedConvFCBBoxHead_arcface(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024,
                s = 30.0,
                m=0.5,
                easy_margin = False,
                loss_ratio=0.02,
                infer_acrface_score = 0.5,
                *args, **kwargs):
        super(RotatedConvFCBBoxHead_arcface, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        # assert self.num_classes == self.num_classes1 + self.num_classes2
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
                # cls_channels2 = self.loss_cls.get_cls_channels(self.num_classes2)# TODO insert
            else:
                cls_channels = self.num_classes + 1
                # cls_channels2 = self.num_classes2 + 1  #TODO insert
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            self.fc_cls2 = ArcMarginProduct(fc_out_channels, self.num_classes, s, m, easy_margin)
            # self.fc_cls2 = build_linear_layer(# TODO inser 
            #     self.cls_predictor_cfg,
            #     in_features=self.cls_last_dim,
            #     out_features=cls_channels2)
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        
        self.criterion = FocalLoss(gamma=2)
        self.loss_ratio = loss_ratio
        self.infer_acrface_score = infer_acrface_score

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.training:
            cls_score = torch.cat([cls_score,x_cls],dim=1)
        else:
            labels_neg =x_cls.new_ones(x_cls.shape[0],1)*self.num_classes
            acrface_logits = self.fc_cls2(x_cls,labels_neg,self.training)
            scores = F.softmax(cls_score, dim=-1)[:,:-1] #不要背景类
            pred_scores, indeces = torch.max(scores,dim=-1)
            pos = pred_scores>self.infer_acrface_score
            cls_score[pos][:,:-1] = acrface_logits[pos]
            cls_score[pos][:,-1]=-1000 
        return cls_score, bbox_pred


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                cls_score1_all = cls_score[:,:self.num_classes+1]
                cls_feature = cls_score[:,self.num_classes+1:]
                index_foreground = labels<self.num_classes
                labels_foreground = labels[index_foreground]
                cls_feature_foreground = cls_feature[index_foreground]
                # label_weights_foreground = label_weights[index_foreground]
                loss_cls1_ = self.loss_cls(
                    cls_score1_all,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                
                acrface_logits = self.fc_cls2(cls_feature_foreground,labels_foreground)
                loss_cls2_ = self.criterion(acrface_logits, labels_foreground)*self.loss_ratio
                # loss_cls2_ = self.loss_cls(
                #     acrface_logits,
                #     labels_foreground,
                #     label_weights_foreground,
                #     avg_factor=avg_factor,
                #     reduction_override=reduction_override)*0.5
                
                if isinstance(loss_cls1_, dict):
                    losses.update(loss_cls1_)
                    losses.update(loss_cls2_)
                else:
                    losses['loss_cls'] = loss_cls1_
                    losses['loss_face'] = loss_cls2_
                if self.custom_activation:
                    acc1_ = self.loss_cls.get_accuracy(cls_score1_all, labels)
                    losses.update(acc1_)
                    acc2_ = self.loss_cls.get_accuracy(acrface_logits, labels_foreground)
                    losses.update(acc2_)
                else:
                    losses['acc'] = accuracy(cls_score1_all, labels)
                    losses['acc_face'] = accuracy(acrface_logits, labels_foreground)
                
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
    

@ROTATED_HEADS.register_module()
class RotatedConvFCBBoxHead_bpfc(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024,
                # s = 30.0,
                # m=0.5,
                # easy_margin = False,
                loss_pfc_weight=0.1,
                margin_list = (1.0, 0.0, 0.4),
                sample_rate = 0.3,
                interclass_filtering_threshold=0.4,
                *args, **kwargs):
        super(RotatedConvFCBBoxHead_bpfc, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        # assert self.num_classes == self.num_classes1 + self.num_classes2
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        self.margin_list = margin_list
        self.sample_rate = sample_rate
        self.interclass_filtering_threshold = interclass_filtering_threshold
        # self.filter_score = filter_score
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
                # cls_channels2 = self.loss_cls.get_cls_channels(self.num_classes2)# TODO insert
            else:
                cls_channels = self.num_classes + 1
                cls_channels = 2
                # cls_channels2 = self.num_classes2 + 1  #TODO insert
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            # self.fc_cls2 = ArcMarginProduct(fc_out_channels, self.num_classes, s, m, easy_margin)
            try:
                world_size = int(os.environ["WORLD_SIZE"])
                rank = int(os.environ["RANK"])
                distributed.init_process_group("nccl")
            except KeyError:
                world_size = 1
                rank = 0
                distributed.init_process_group(
                    backend="nccl",
                    init_method="tcp://127.0.0.1:12584",
                    rank=rank,
                    world_size=world_size,
                )
            margin_loss = CombinedMarginLoss(
                64,
                self.margin_list[0],
                self.margin_list[1],
                self.margin_list[2],
                self.interclass_filtering_threshold)
            self.fc_cls2 = PartialFC_V2(
                margin_loss, fc_out_channels, self.num_classes,
                self.sample_rate, False)
            self.fc_cls2.train().cuda()
            
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        
        self.criterion = FocalLoss(gamma=2)
        self.loss_ratio = loss_pfc_weight
        # self.infer_acrface_score = infer_acrface_score

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.training:
            cls_score = torch.cat([cls_score,x_cls],dim=1)
        else:
            labels_neg = x_cls.new_zeros(x_cls.shape[0],1)
            pfc_logits, pfc_labels = self.fc_cls2(x_cls,labels_neg,self.training)
            cls_score = torch.hstack((cls_score,cls_score.new_tensor(pfc_logits)))
        return cls_score, bbox_pred

 
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                # cls_score1 = cls_score[:,:self.num_classes+1]
                # cls_feature = cls_score[:,self.num_classes+1:]
                cls_score1 = cls_score[:,:2]
                cls_feature = cls_score[:,2:]
                
                #获取前景类
                index_foreground = labels<self.num_classes
                labels_b = index_foreground*1
                labels_foreground = labels[index_foreground]
                cls_feature_foreground = cls_feature[index_foreground]
                # #过滤错误比较明显的类
                # scores = F.softmax(cls_score1, dim=-1) 
                # pred_scores, indeces = torch.max(scores,dim=-1)
                # index_wrong = (indeces != labels) & (pred_scores>self.filter_score) 
                # index_filter = ~index_wrong
                # cls_score1 = cls_score1[index_filter]
                # labels1 = labels[index_filter]
                # label_weights = label_weights[index_filter]
                
                loss_cls1_ = self.loss_cls(
                    cls_score1,
                    labels_b,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                
                pfc_logits,pfc_labels = self.fc_cls2(cls_feature_foreground,labels_foreground)
                # ind = labels==self.num_classes
                # labels[ind]=-1
                # pfc_logits,pfc_labels = self.fc_cls2(cls_feature,labels)
                # loss_cls2_ = self.fc_cls2.dist_cross_entropy(pfc_logits, pfc_labels)*self.loss_ratio
                #之后在加大分类 边界，然后再加过滤
                loss_cls2_ = F.cross_entropy(pfc_logits,pfc_labels.reshape(-1))*self.loss_ratio
                if isinstance(loss_cls1_, dict):
                    losses.update(loss_cls1_)
                    losses.update(loss_cls2_)
                else:
                    losses['loss_cls'] = loss_cls1_
                    losses['loss_face'] = loss_cls2_
                if self.custom_activation:
                    acc1_ = self.loss_cls.get_accuracy(cls_score1, labels_b)
                    losses.update(acc1_)
                    acc2_ = self.loss_cls.get_accuracy(pfc_logits, pfc_labels)
                    losses.update(acc2_)
                else:
                    losses['acc'] = accuracy(cls_score1, labels_b )
                    losses['acc_face'] = accuracy(pfc_logits, pfc_labels.reshape(-1))
                    # losses['filter']=index_wrong.sum()*100/index_wrong.shape[0]
                    
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 6) and last
                dimension 6 represent (cx, cy, w, h, a, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            #将二分类分数与人脸分类分数分开，并分别计算softmax
            scores_b = cls_score[:,:2]
            scores_pfc = cls_score[:,2:]
            scores_b = F.softmax( scores_b, dim=-1) if cls_score is not None else None
            scores_pfc = F.softmax( scores_pfc, dim=-1) if cls_score is not None else None
            
            # #根据arc的计算公式计算类别数
            # logits = scores_pfc
            # max_logits, _ = torch.max(logits, dim=1, keepdim=True)
            # logits.sub_(max_logits)
            # logits.exp_()
            # sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
            # logits.div_(sum_logits_exp)
        
            #将前景概率乘上 人脸分类概率，然后将背景概率作为新的背景概率
            scores_pfc_b = scores_pfc * scores_b[:,0:1]
            scores = torch.hstack((scores_pfc_b,scores_b[:,1:]))
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = bboxes.view(bboxes.size(0), -1, 5)
            bboxes[..., :4] = bboxes[..., :4] / scale_factor
            bboxes = bboxes.view(bboxes.size(0), -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms_rotated(
                bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels
    
@ROTATED_HEADS.register_module()
class RotatedConvFCBBoxHead_pfc(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024,
                # s = 30.0,
                # m=0.5,
                # easy_margin = False,
                loss_ratio=0.02,
                margin_list = (1.0, 0.0, 0.4),
                sample_rate = 0.3,
                interclass_filtering_threshold=0.4,
                filter_score = 0.95,
                infer_acrface_score = 0.5,
                *args, **kwargs):
        super(RotatedConvFCBBoxHead_pfc, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        # assert self.num_classes == self.num_classes1 + self.num_classes2
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        self.margin_list = margin_list
        self.sample_rate = sample_rate
        self.interclass_filtering_threshold = interclass_filtering_threshold
        self.filter_score = filter_score
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
                # cls_channels2 = self.loss_cls.get_cls_channels(self.num_classes2)# TODO insert
            else:
                cls_channels = self.num_classes + 1
                # cls_channels2 = self.num_classes2 + 1  #TODO insert
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            # self.fc_cls2 = ArcMarginProduct(fc_out_channels, self.num_classes, s, m, easy_margin)
            try:
                world_size = int(os.environ["WORLD_SIZE"])
                rank = int(os.environ["RANK"])
                distributed.init_process_group("nccl")
            except KeyError:
                world_size = 1
                rank = 0
                distributed.init_process_group(
                    backend="nccl",
                    init_method="tcp://127.0.0.1:12584",
                    rank=rank,
                    world_size=world_size,
                )
            margin_loss = CombinedMarginLoss(
                64,
                self.margin_list[0],
                self.margin_list[1],
                self.margin_list[2],
                self.interclass_filtering_threshold)
            self.fc_cls2 = PartialFC_V2(
                margin_loss, fc_out_channels, self.num_classes,
                self.sample_rate, False)
            self.fc_cls2.train().cuda()
            
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        
        self.criterion = FocalLoss(gamma=2)
        self.loss_ratio = loss_ratio
        self.infer_acrface_score = infer_acrface_score

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.training:
            cls_score = torch.cat([cls_score,x_cls],dim=1)
            return cls_score, bbox_pred
        else:
            labels_neg =x_cls.new_ones(x_cls.shape[0],1)*self.num_classes
            acrface_logits = self.fc_cls2(x_cls,labels_neg,self.training)
            scores = F.softmax(cls_score, dim=-1)[:,:-1] #不要背景类
            pred_scores, _ = torch.max(scores,dim=-1)
            pos = pred_scores > self.infer_acrface_score
            cls_score_new = torch.zeros_like(cls_score).float()
            cls_score_new[~pos] = cls_score[~pos].float()
            new_acrface_logits = torch.hstack((acrface_logits[0],acrface_logits[0][:,0:1]))
            new_acrface_logits[:,-1:] = -1000 
            cls_score_new[pos] = new_acrface_logits[pos]
            return cls_score_new, bbox_pred

 
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                cls_score1 = cls_score[:,:self.num_classes+1]
                cls_feature = cls_score[:,self.num_classes+1:]
                index_foreground = labels<self.num_classes
                labels_foreground = labels[index_foreground]
                cls_feature_foreground = cls_feature[index_foreground]
                # label_weights_foreground = label_weights[index_foreground]
                scores = F.softmax(cls_score1, dim=-1) 
                pred_scores, indeces = torch.max(scores,dim=-1)
                index_wrong = (indeces != labels) & (pred_scores>self.filter_score) 
                index_filter = ~index_wrong
                cls_score1 = cls_score1[index_filter]
                labels1 = labels[index_filter]
                label_weights = label_weights[index_filter]
                loss_cls1_ = self.loss_cls(
                    cls_score1,
                    labels1,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                
                pfc_logits,pfc_labels = self.fc_cls2(cls_feature_foreground,labels_foreground)
                loss_cls2_ = self.fc_cls2.dist_cross_entropy(pfc_logits, pfc_labels)*self.loss_ratio
           
                
                if isinstance(loss_cls1_, dict):
                    losses.update(loss_cls1_)
                    losses.update(loss_cls2_)
                else:
                    losses['loss_cls'] = loss_cls1_
                    losses['loss_face'] = loss_cls2_
                if self.custom_activation:
                    acc1_ = self.loss_cls.get_accuracy(cls_score1, labels1)
                    losses.update(acc1_)
                    acc2_ = self.loss_cls.get_accuracy(pfc_logits, pfc_labels)
                    losses.update(acc2_)
                else:
                    losses['acc'] = accuracy(cls_score1, labels1)
                    losses['acc_face'] = accuracy(pfc_logits, pfc_labels.reshape(-1))
                    losses['filter']=index_wrong.sum()*100/index_wrong.shape[0]
                    
                
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
    

@ROTATED_HEADS.register_module()
class RotatedConvFCBBoxHead_pfcc(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024,
                sample_rate = 0.3,
                filter_score = 0.95,
                *args, **kwargs):
        super(RotatedConvFCBBoxHead_pfcc, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        # assert self.num_classes == self.num_classes1 + self.num_classes2
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # self.margin_list = margin_list
        self.sample_rate = sample_rate
        # self.interclass_filtering_threshold = interclass_filtering_threshold
        self.filter_score = filter_score
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
                # cls_channels2 = self.loss_cls.get_cls_channels(self.num_classes2)# TODO insert
            else:
                cls_channels = self.num_classes + 1
                # cls_channels2 = self.num_classes2 + 1  #TODO insert
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)

            self.num_sample  = int((self.num_classes+1)*self.sample_rate)
            
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)


    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return x_cls, bbox_pred
    
    def sample(self, weight, labels, index_positive):
        """
            This functions will change the value of labels
            Parameters:
            -----------
            labels: torch.Tensor
                pass
            index_positive: torch.Tensor
                pass
            optimizer: torch.optim.Optimizer
                pass
        """
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).cuda()
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_classes]).cuda()
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1].cuda()
                index = index.sort()[0].cuda()
            else:
                index = positive
            self.weight_index = index
            labels_pfc = labels.clone()
            labels_pfc[index_positive] = torch.searchsorted(index, labels[index_positive])
        return weight[self.weight_index], labels_pfc
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                # #过滤错误的标签
                pfc_logits,_  = self.fc_cls(cls_score, labels, is_train=False)
                scores = F.softmax(pfc_logits, dim=-1) 
                pred_scores, indeces = torch.max(scores,dim=-1)
                index_wrong = (indeces != labels) & (pred_scores>self.filter_score) 
                index_filter = ~index_wrong
                cls_score = cls_score[index_filter]
                labels_filter = labels[index_filter]
                label_weights = label_weights[index_filter]
                
                pfc_logits,pfc_labels= self.fc_cls2(cls_score, labels_filter,is_train=True)
                
                # loss_cls1_ = self.fc_cls2.dist_cross_entropy(pfc_logits, pfc_labels)*self.loss_pfc_weight
                # loss_cls1_ = F.cross_entropy(pfc_logits,pfc_labels.reshape(-1))*self.loss_pfc_weight
                loss_cls1_ = self.loss_cls(
                    pfc_logits,
                    pfc_labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls1_, dict):
                    losses.update(loss_cls1_)
                else:
                    losses['loss_cls'] = loss_cls1_
                    # losses['loss_face'] = loss_cls2_
                if self.custom_activation:
                    acc1_ = self.loss_cls.get_accuracy(pfc_logits, pfc_labels.reshape(-1))
                    losses.update(acc1_)
                else:
                    losses['acc'] = accuracy(pfc_logits, pfc_labels.reshape(-1))
                    losses['filter']=index_wrong.sum()*100/index_wrong.shape[0]
                    
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 6) and last
                dimension 6 represent (cx, cy, w, h, a, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            #将二分类分数与人脸分类分数分开，并分别计算softmax
            labels = cls_score[:,0].int()
            pfc_logits,_= self.fc_cls2(cls_score,labels,is_train=False)
            scores = F.softmax( pfc_logits, dim=-1) if cls_score is not None else None
            # pfc_embeddings,pfc_weight, _ = self.fc_cls2(cls_score,labels=None,training=False)
            # pfc_logits = linear(pfc_embeddings, pfc_weight)
            # max_logits, _ = torch.max(pfc_logits, dim=1, keepdim=True)
            # pfc_logits.sub_(max_logits)
            # pfc_logits.exp_()
            # sum_logits_exp = torch.sum(pfc_logits, dim=1, keepdim=True)
            # pfc_logits.div_(sum_logits_exp)
            # scores = F.softmax( pfc_logits, dim=-1) if cls_score is not None else None
        
            # #将前景概率乘上 人脸分类概率，然后将背景概率作为新的背景概率
            # scores_pfc_b = scores_pfc * scores_b[:,0:1]
            # scores = torch.hstack((scores_pfc_b,scores_b[:,1:]))
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = bboxes.view(bboxes.size(0), -1, 5)
            bboxes[..., :4] = bboxes[..., :4] / scale_factor
            bboxes = bboxes.view(bboxes.size(0), -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms_rotated(
                bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels
        
        
@ROTATED_HEADS.register_module()
class RotatedShared2FCBBoxHead_double(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024,
                num_classes1=80,
                num_classes2=80,
                filter_score=0.8,
                flip_ratio=0.1,
                *args, **kwargs):
        super(RotatedShared2FCBBoxHead_double, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        assert self.num_classes == self.num_classes1 + self.num_classes2
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes1)
                cls_channels2 = self.loss_cls.get_cls_channels(self.num_classes2)# TODO insert
            else:
                cls_channels = self.num_classes1 + 1
                cls_channels2 = self.num_classes2 + 1  #TODO insert
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            self.fc_cls2 = build_linear_layer(# TODO inser 
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels2)
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)
            
        self.filter_score=filter_score
        self.flip_ratio=flip_ratio


    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        cls_score2 = self.fc_cls2(x_cls) if self.with_cls else None # TODO insert
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.training:
            cls_score = torch.cat([cls_score,cls_score2],dim=1)
        return cls_score, bbox_pred


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:

                index1 = labels<self.num_classes1
                index2 = (labels>=self.num_classes1) & (labels < self.num_classes)
                indexb = labels==self.num_classes
                label_b1 = labels[indexb]-self.num_classes2
                label_b2 = labels[indexb]-self.num_classes1
                labels1 = labels[index1]
                labels2 = labels[index2]-self.num_classes1

                cls_score1_all = cls_score[:,:self.num_classes1+1]
                cls_score2_all = cls_score[:,self.num_classes1+1:]
                cls_score1 = cls_score1_all[index1]
                cls_score2 = cls_score2_all[index2]
                cls_score1_back = cls_score1_all[indexb]
                cls_score2_back = cls_score2_all[indexb]

                scores = F.softmax(cls_score1, dim=-1) 
                pred_scores, indeces = torch.max(scores,dim=-1)
                index_wrong = (indeces != labels1) & (pred_scores>self.filter_score) 
                index_filter = ~index_wrong
                cls_score1_filter = cls_score1[index_filter]
                labels1_filter = labels1[index_filter]

                cls_score1 = torch.cat([cls_score1_filter,cls_score1_back],dim=0)
                cls_score2 = torch.cat([cls_score2,cls_score2_back],dim=0)

                labels1 = torch.cat([labels1_filter,label_b1])
                labels2 = torch.cat([labels2,label_b2])
                
                label_weights1 = label_weights[index1]
                label_weights1_filter = label_weights1[index_filter]
                label_weights2 = label_weights[index2]
                label_weightsb = label_weights[indexb]
                label_weights1 = torch.cat([label_weights1_filter,label_weightsb])
                label_weights2 = torch.cat([label_weights2,label_weightsb])

                loss_cls1_ = self.loss_cls(
                    cls_score1,
                    labels1,
                    label_weights1,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)*0.5
                loss_cls2_ = self.loss_cls(
                    cls_score2,
                    labels2,
                    label_weights2,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)*0.5
                if isinstance(loss_cls1_, dict):
                    losses.update(loss_cls1_)
                    losses.update(loss_cls2_)
                else:
                    losses['loss_cls1'] = loss_cls1_
                    losses['loss_cls2'] = loss_cls2_
                if self.custom_activation:
                    acc1_ = self.loss_cls.get_accuracy(cls_score1, labels1)
                    losses.update(acc1_)
                    acc2_ = self.loss_cls.get_accuracy(cls_score2, labels2)
                    losses.update(acc2_)
                else:
                    losses['acc1'] = accuracy(cls_score1, labels1)
                    losses['acc2'] = accuracy(cls_score2, labels2)
                    losses['filter']=index_wrong.sum()*100/index_wrong.shape[0]
                
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


@ROTATED_HEADS.register_module()
class RotatedShared2FCBBoxHead_treble(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024,
                num_classes1=98,
                num_classes2=39,
                num_classes3=231,
                filter_score=0.9,
                *args, **kwargs):
        super(RotatedShared2FCBBoxHead_treble, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3
        self.filter_score = filter_score
        assert self.num_classes == self.num_classes1 + self.num_classes2 + self.num_classes3
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes1)
                cls_channels2 = self.loss_cls.get_cls_channels(self.num_classes2)# TODO insert
                cls_channels3 = self.loss_cls.get_cls_channels(self.num_classes3)# TODO insert
            else:
                cls_channels = self.num_classes1 + 1
                cls_channels2 = self.num_classes2 + 1  #TODO insert
                cls_channels3 = self.num_classes3 + 1  #TODO insert
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            self.fc_cls2 = build_linear_layer(# TODO inser 
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels2)
            self.fc_cls3 = build_linear_layer(# TODO inser 
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels3)
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)


    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        cls_score2 = self.fc_cls2(x_cls) if self.with_cls else None # TODO insert
        cls_score3 = self.fc_cls3(x_cls) if self.with_cls else None # TODO insert
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.training:
            cls_score = torch.cat([cls_score,cls_score2,cls_score3],dim=1)
        return cls_score, bbox_pred


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                #根据标签大小，分离要训练的样本
                index1 = labels<self.num_classes1
                index2 = (labels>=self.num_classes1) & (labels < self.num_classes1+self.num_classes2)
                index3 = (labels>= self.num_classes1+self.num_classes2) & (labels < self.num_classes)

                #3将背景类变为每个分类头的类别数量
                indexb = labels==self.num_classes
                label_b1 = labels[indexb]-self.num_classes2 - self.num_classes3
                label_b2 = labels[indexb]-self.num_classes3 - self.num_classes1
                label_b3 = labels[indexb]-self.num_classes1 - self.num_classes2
                #范围变为0-numclass1
                labels1 = labels[index1]
                labels2 = labels[index2] - self.num_classes1
                labels3 = labels[index3] - self.num_classes1 - self.num_classes2

                #根据相关索引取出训练样本 标签 权重
                cls_score1_all = cls_score[:,:self.num_classes1+1]
                cls_score2_all = cls_score[:,self.num_classes1+1:self.num_classes1+self.num_classes2+2]
                cls_score3_all = cls_score[:,self.num_classes1+self.num_classes2+2:]
                cls_score1 = cls_score1_all[index1]
                cls_score2 = cls_score2_all[index2]
                cls_score3 = cls_score3_all[index3]
                cls_score1_back = cls_score1_all[indexb]
                cls_score2_back = cls_score2_all[indexb]
                cls_score3_back = cls_score3_all[indexb]
                label_weights1 = label_weights[index1]
                label_weights2 = label_weights[index2]
                label_weights3 = label_weights[index3]
                label_weightsb = label_weights[indexb]

                #过滤可能是错误的训练样本
                scores = F.softmax(cls_score1, dim=-1) 
                pred_scores, indeces = torch.max(scores,dim=-1)
                index_wrong = (indeces != labels1) & (pred_scores>self.filter_score) 
                index_filter = ~index_wrong
                cls_score1 = cls_score1[index_filter]
                labels1 = labels1[index_filter]
                label_weights1 = label_weights1[index_filter]

                #将前景背景拼接起来
                labels1 = torch.cat([labels1,label_b1])
                labels2 = torch.cat([labels2,label_b2])
                labels3 = torch.cat([labels3,label_b3])
                cls_score1 = torch.cat([cls_score1,cls_score1_back],dim=0)
                cls_score2 = torch.cat([cls_score2,cls_score2_back],dim=0)
                cls_score3 = torch.cat([cls_score3,cls_score3_back],dim=0)
                label_weights1 = torch.cat([label_weights1,label_weightsb])
                label_weights2 = torch.cat([label_weights2,label_weightsb])
                label_weights3 = torch.cat([label_weights3,label_weightsb])

                loss_cls1_ = self.loss_cls(
                    cls_score1,
                    labels1,
                    label_weights1,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)*0.4
                loss_cls2_ = self.loss_cls(
                    cls_score2,
                    labels2,
                    label_weights2,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)*0.3
                loss_cls3_ = self.loss_cls(
                    cls_score3,
                    labels3,
                    label_weights3,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)*0.3
                if isinstance(loss_cls1_, dict):
                    losses.update(loss_cls1_)
                    losses.update(loss_cls2_)
                    losses.update(loss_cls3_)
                else:
                    losses['loss_cls1'] = loss_cls1_
                    losses['loss_cls2'] = loss_cls2_
                    losses['loss_cls3'] = loss_cls3_
                if self.custom_activation:
                    acc1_ = self.loss_cls.get_accuracy(cls_score1, labels1)
                    losses.update(acc1_)
                    acc2_ = self.loss_cls.get_accuracy(cls_score2, labels2)
                    losses.update(acc2_)
                    acc3_ = self.loss_cls.get_accuracy(cls_score3, labels3)
                    losses.update(acc3_)
                else:
                    losses['acc1'] = accuracy(cls_score1, labels1)
                    losses['acc2'] = accuracy(cls_score2, labels2)
                    losses['acc3'] = accuracy(cls_score3, labels3)
                    # if index_wrong.shape[0]==0:
                    #     losses['filter']=index_wrong.sum()
                    # else:
                    losses['filter']=index_wrong.sum()*100/index_wrong.shape[0]

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


@ROTATED_HEADS.register_module()
class RotatedShared2FCBBoxHead_double_filter(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024,
                num_classes1=80,
                num_classes2=80,
                *args, **kwargs):
        super(RotatedShared2FCBBoxHead_double_filter, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        assert self.num_classes == self.num_classes1 + self.num_classes2
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes1)
                cls_channels2 = self.loss_cls.get_cls_channels(self.num_classes2)# TODO insert
            else:
                cls_channels = self.num_classes1 + 1
                cls_channels2 = self.num_classes2 + 1  #TODO insert
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
            self.fc_cls2 = build_linear_layer(# TODO inser 
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels2)
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)


    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        cls_score2 = self.fc_cls2(x_cls) if self.with_cls else None # TODO insert
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.training:
            cls_score = torch.cat([cls_score,cls_score2],dim=1)
        return cls_score, bbox_pred


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:

                index1 = labels<self.num_classes1+1
                index2 = (labels>self.num_classes1) & (labels < self.num_classes1+self.num_classes2)
                indexb = labels==self.num_classes
                label_b1 = labels[indexb]-self.num_classes2
                label_b2 = labels[indexb]-self.num_classes1
                labels1 = labels[index1]
                labels2 = labels[index2]-self.num_classes1

                labels1 = torch.cat([labels1,label_b1])
                labels2 = torch.cat([labels2,label_b2])
                
                cls_score1_all = cls_score[:,:self.num_classes1+1]
                cls_score2_all = cls_score[:,self.num_classes1+1:]
                cls_score1 = cls_score1_all[index1]
                cls_score2 = cls_score2_all[index2]
                cls_score1_back = cls_score1_all[indexb]
                cls_score2_back = cls_score2_all[indexb]

                cls_score1 = torch.cat([cls_score1,cls_score1_back],dim=0)
                cls_score2 = torch.cat([cls_score2,cls_score2_back],dim=0)

                label_weights1 = label_weights[index1]
                label_weights2 = label_weights[index2]
                label_weightsb = label_weights[indexb]
                label_weights1 = torch.cat([label_weights1,label_weightsb])
                label_weights2 = torch.cat([label_weights2,label_weightsb])

                loss_cls1_ = self.loss_cls(
                    cls_score1,
                    labels1,
                    label_weights1,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)*0.5
                loss_cls2_ = self.loss_cls(
                    cls_score2,
                    labels2,
                    label_weights2,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)*0.5
                if isinstance(loss_cls1_, dict):
                    losses.update(loss_cls1_)
                    losses.update(loss_cls2_)
                else:
                    losses['loss_cls1'] = loss_cls1_
                    losses['loss_cls2'] = loss_cls2_
                if self.custom_activation:
                    acc1_ = self.loss_cls.get_accuracy(cls_score1, labels1)
                    losses.update(acc1_)
                    acc2_ = self.loss_cls.get_accuracy(cls_score2, labels2)
                    losses.update(acc2_)
                else:
                    losses['acc1'] = accuracy(cls_score1, labels1)
                    losses['acc2'] = accuracy(cls_score2, labels2)
                
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


@ROTATED_HEADS.register_module()
class RotatedShared2FCBBoxHead(RotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RotatedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

@ROTATED_HEADS.register_module()
class RotatedKFIoUShared2FCBBoxHead(RotatedConvFCBBoxHead):
    """KFIoU RoI head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RotatedKFIoUShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function."""
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                bbox_pred_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_pred)
                bbox_targets_decode = self.bbox_coder.decode(
                    rois[:, 1:], bbox_targets)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    pred_decode=pos_bbox_pred_decode,
                    targets_decode=bbox_targets_decode[pos_inds.type(
                        torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
