import mmcv
from mmcv import Config
import cv2
from mmrotate.datasets import build_dataset
from mmdet.datasets import build_dataloader
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)
import math
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
from pycocotools.coco import COCO
# from mmdet.core import rotated_box_to_poly_single,gt_mask_bp_obbs_list

def rotate_rect2cv(rotatebox):
    #此程序将rotatexml中旋转矩形的表示，转换为cv2的RotateRect
    [x_center,y_center,w,h,angle]=rotatebox[0:5]
    angle_mod=angle*180/np.pi%180
    if angle_mod>=0 and angle_mod<90:
        [cv_w,cv_h,cv_angle]=[h,w,angle_mod-90]
    if angle_mod>=90 and angle_mod<180:
        [cv_w,cv_h,cv_angle]=[w,h,angle_mod-180]
    return ((x_center,y_center),(cv_w,cv_h),cv_angle)


def showAnns( rootdir,imgname, img, rboxes,labels, CLASSES,colors):
    # r = 5
    rboxes=rboxes.reshape(-1,8)
    # print('show: {}'.format( len(rboxes)))
    for i in range(rboxes.shape[0]):
        pts=rboxes[i]#[4:12]

        label=labels[i]
        class_name=CLASSES[label]
        color=np.array(np.clip(colors[label],1,255),np.int32)
        color = (int(color[0]),int(color[1]),int(color[2]))
        pts =np.array(pts.reshape((-1, 1, 2)), np.int32)
        center=np.mean(pts.reshape(4,2),axis=0)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        cv2.circle(img,( int(center[0]),int(center[1])),2,color=color)
        text='{:s}'.format(class_name)
        cv2.putText(img, text, (int(center[0]), int(center[1] )), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(os.path.join(rootdir, imgname),img)
    # plt.savefig(os.path.join(rootdir, imgname), bbox_inches="tight", pad_inches=0.0)
    # plt.close()


def run_datatloader(cfg):
    """
    可视化数据增强后的效果，同时也可以确认训练样本是否正确
    Args:
        cfg: 配置
    Returns:
    """
    # Build dataset
    cfg.data.samples_per_gpu=1
    dataset = build_dataset(cfg.data.train)
    # prepare data loaders
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist= False,
            shuffle=False)
    CLASSES=dataset.CLASSES
    outdir=os.path.join(cfg.data_root,'showdataloder')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # colors=[]
    # num_class=cfg.model.bbox_head.num_classes
    # for i in range(num_class):
    #     c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
    #     colors.append(c)
    color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            # 0.300, 0.300, 0.300,
            # 0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            # 1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            # 0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]

    for i, data_batch in enumerate(data_loader):
        img_batch =data_batch['img'].data
        filename=data_batch['img_metas'].data[0][0]['ori_filename']
        gt_label = data_batch['gt_labels'].data
        gt_box = data_batch['gt_bboxes'].data
        gt_masks=data_batch['gt_polygons'].data
        print(filename)
        for batch_i in range(len(img_batch)):
            img = img_batch[batch_i]
            mean_value = np.array(cfg.img_norm_cfg['mean'])
            std_value = np.array(cfg.img_norm_cfg['std'])
            img_hwc = np.transpose(np.squeeze(img.numpy()), [1, 2, 0])
            img = (img_hwc * std_value) + mean_value
            img = np.array(img, np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img_numpy_uint8 = np.array(img, np.uint8)
            labels = gt_label[batch_i][0].cpu().numpy()
            # boxes = gt_box[batch_i][0].cpu().numpy()
            # gt_masks = gt_mask_bp_obbs_list(gt_masks)
            gt_mask_arrays=[]
            for i in  gt_masks:
                gt_mask_array=np.array(i[0].masks).squeeze()
                # gt_mask_array = gt_mask_array.astype(float)  # numpy强制类型转换
                gt_mask_arrays.append(gt_mask_array)
            gt_masks=gt_mask_arrays[0]
            # segmap=gt_mask[batch_i][0].cpu().numpy()
            # showAnns(outdir, filename, img, boxes[...,8:16],colors) 
            showAnns(outdir, filename, img, gt_masks,labels, CLASSES,colors) 

if __name__ == '__main__':
    cfg = Config.fromfile(
        '/home/zf/Large-Selective-Kernel-Network/configs/lsknet/lsk_t_fpn_1x_yami_le90_fp16.py')
    run_datatloader(cfg)