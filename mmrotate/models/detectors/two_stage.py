# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
import numpy as np
import mmcv
import torchvision
import cv2
import torch.nn.functional as F

def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
    keep = []#保留框的结果集合
    order = scores.argsort()[::-1]#对检测结果得分进行降序排序
    num = boxes.shape[0]#获取检测框的个数

    suppressed = np.zeros((num), dtype=np.int)
    for _i in range(num):
        if len(keep) >= max_output_size:#若当前保留框集合中的个数大于max_output_size时，直接返回
            break

        i = order[_i]
        if suppressed[i] == 1:#对于抑制的检测框直接跳过
            continue
        keep.append(i)#保留当前框的索引
        # (midx,midy),(width,height), angle)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4]) 
#        r1 = ((boxes[i, 1], boxes[i, 0]), (boxes[i, 3], boxes[i, 2]), boxes[i, 4]) #根据box信息组合成opencv中的旋转bbox
#        print("r1:{}".format(r1))
        area_r1 = boxes[i, 2] * boxes[i, 3]#计算当前检测框的面积
        for _j in range(_i + 1, num):#对剩余的而进行遍历
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]#求两个旋转矩形的交集，并返回相交的点集合
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)#求点集的凸边形
                int_area = cv2.contourArea(order_pts)#计算当前点集合组成的凸边形的面积
                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 0.0000001)

            if inter >= iou_threshold:#对大于设定阈值的检测框进行滤除
                suppressed[j] = 1
    return np.array(keep, np.int64)

@ROTATED_DETECTORS.register_module()
class RotatedTwoStageDetector(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedTwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 5).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    # def simple_test(self, img, img_metas, proposals=None, rescale=False):
    #     """Test without augmentation."""

    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #     x = self.extract_feat(img)
    #     if proposals is None:
    #         proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
    #     else:
    #         proposal_list = proposals

    #     return self.roi_head.simple_test(
    #         x, proposal_list, img_metas, rescale=rescale)


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)


    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        cfg = self.test_cfg 
        aug_test=cfg.get('aug_test', False)
        if not aug_test:
            assert self.with_bbox, 'Bbox head must be implemented.'
            x = self.extract_feat(img)
            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            result = self.roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)
            return result
        else:
            #1原始测试
            x = self.extract_feat(img)
            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            result = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
            
            #2水平翻转
            transform = torchvision.transforms.RandomHorizontalFlip(p=1.0)
            aug_img = transform(img.clone())
            x = self.extract_feat(aug_img)
            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            result_horizontal = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
          
            #3垂直翻转
            transform = torchvision.transforms.RandomVerticalFlip(p=1.0)
            aug_img = transform(img.clone())
            x = self.extract_feat(aug_img)
            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            result_vertical = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
            
            #4对角翻转 其实是先水平翻转 在垂直翻转，相当于旋转180度，
            transform = torchvision.transforms.RandomHorizontalFlip(p=1.0)
            aug_img = transform(aug_img.clone())
            x = self.extract_feat(aug_img)
            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            result_diagonal = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

            # 5#旋转90度
            # aug_img = self.rot_img(img.clone(),90)
            # x = self.extract_feat(aug_img)
            # if proposals is None:
            #     proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            # else:
            #     proposal_list = proposals
            # result_90 = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
          
            # #6旋转270度
            # aug_img = self.rot_img(img.clone(),270)
            # x = self.extract_feat(aug_img)
            # if proposals is None:
            #     proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            # else:
            #     proposal_list = proposals
            # result_270 = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
          

            #开始处理得到的结果
            h, w, _ = img_metas[0]['ori_shape']
            # h, w =img.shape[-2],img.shape[-1]
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
            
            #2horizontal flip
            result_horizontal_ok = result_horizontal.copy()
            for ind,res in enumerate(result_horizontal[0]):
                if res.size>0:
                    # cx,cy,w,h,angle,score = res
                    result_horizontal_ok[0][ind][:,0] = w-1-res[:,0]
                    result_horizontal_ok[0][ind][:,4] = (np.pi - res[:,4]) % np.pi 
            
            #3 vertical flip
            result_vertical_ok = result_vertical.copy()
            for ind,res in enumerate(result_vertical[0]):
                if res.size>0:
                    # cx,cy,w,h,angle,score = res
                    result_vertical_ok[0][ind][:,1] = h - 1 - res[:,1]
                    result_vertical_ok[0][ind][:,4] =  - res[:,4]
                    
            #4 rotate 180 也就是 先水平后垂直翻转        
            result_diagonal_ok = result_diagonal.copy()
            for ind,res in enumerate(result_diagonal[0]):
                if res.size>0:
                    result_diagonal_ok[0][ind][:,0] = w - 1 - res[:,0]
                    result_diagonal_ok[0][ind][:,1] = h - 1 - res[:,1]  
            
            # #5 rotate 90        
            # result_90_ok = result_90.copy()
            # for ind,res in enumerate(result_90[0]):
            #     if res.size>0:
            #         result_90_ok[0][ind] = self.rot_boxes(result_90_ok[0][ind], 90,center)
            
            # #6 rotate 270
            # result_270_ok = result_270.copy()
            # for ind,res in enumerate(result_270[0]):
            #     if res.size>0:
            #         result_270_ok[0][ind] = self.rot_boxes(result_270_ok[0][ind], 270,center)
                    
            #fusion 结果融合
            result_aug = result.copy()
            for ind,res in enumerate(result[0]):
                # result_aug[0][ind] = np.concatenate((result[0][ind], result_horizontal_ok[0][ind]),axis=0)
                result_aug[0][ind] = np.concatenate((result[0][ind], result_horizontal_ok[0][ind] ,result_vertical_ok[0][ind] ,result_diagonal_ok[0][ind]),axis=0)
                # result_aug[0][ind] = np.concatenate((result[0][ind], result_horizontal_ok[0][ind] ,\
                #                                 result_vertical_ok[0][ind] ,result_diagonal_ok[0][ind],\
                #                                     result_90_ok[0][ind],result_270_ok[0][ind]),axis=0)
                #这里还要nms
                if result_aug[0][ind].size>0:
                    cvrboxes = result_aug[0][ind][:,:5].copy()
                    cvrboxes[:,4]= cvrboxes[:,4] * 180 / np.pi
                    keep = nms_rotate_cpu(cvrboxes,result_aug[0][ind][:,5],0.15, 400)   #这里可以改
                    result_aug[0][ind] = result_aug[0][ind][keep]
            return result_aug
            # return result_90_ok

    def rot_img(self,x, theta):
        # theta = torch.tensor(theta)theta
        device=x.device
        theta=x.new_tensor(theta* np.pi / (180))#.to(device)#表示这里是反的顺时针旋转 是负号
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]]).to(x.device)
        rot_mat=rot_mat[None, ...]#.repeat(x.shape[0],1,1)[None, ...]
        out_size = torch.Size((1, x.size()[1],  x.size()[2],  x.size()[3]))
        grid = F.affine_grid(rot_mat, out_size)
        rotate = F.grid_sample(x, grid)
        # rotate = F.grid_sample(x.unsqueeze(0).unsqueeze(0), grid)
        return rotate#.squeeze()
    
    def rot_boxes(self,bboxes, theta,center):
        theta = theta/180*np.pi
        bboxes = torch.from_numpy(bboxes)
        scores=bboxes[:,5:6]
        rboxes=bboxes[:,:5]
        N = rboxes.shape[0]
        
        angle=scores.clone()
        angle[:]=theta
        x_ctr, y_ctr,  = rboxes.select(1, 0), rboxes.select( 1, 1)
        new_x_c,new_y_c=x_ctr - center[0], y_ctr - center[1]

        rects = torch.stack([new_x_c, new_y_c], dim=0).reshape(2, 1, N).permute(2, 0, 1)
        sin, cos = torch.sin(angle), torch.cos(angle)
        # M.shape=[N,2,2]
        M = torch.stack([cos, -sin, sin, cos],
                        dim=0).reshape(2, 2, N).permute(2, 0, 1)
        # polys:[N,8]
        polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
        rboxes[:,0] = polys[:,0]+center[0]
        rboxes[:,1] = polys[:,1]+center[1]
        rboxes[:,4]+=theta
        return  torch.cat([rboxes, scores], 1).numpy()