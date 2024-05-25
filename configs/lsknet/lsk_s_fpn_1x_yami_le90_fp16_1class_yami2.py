_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le90'
gpu_number = 1
# fp16 = dict(loss_scale='dynamic')
model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='LSKNet',
        embed_dims=[64, 128, 320, 512],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2,2,4,2],
        init_cfg=dict(type='Pretrained', checkpoint="./checkpoints/lsk_s_backbone-e9d2e551.pth"),
        # norm_cfg=dict(type='SyncBN', requires_grad=True)),
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=dict(
        type='FPN',
        # in_channels=[32, 64, 160, 256],
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                gpu_assign_thr=800,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,#changeTrue
                iou_calculator=dict(type='RBboxOverlaps2D'),
                gpu_assign_thr=800,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))

# dataset settings
dataset_type = 'Yami1cDataset'
data_root = "/home/zf/Dataset/0yami8k/" 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        # rect_classes=[9, 11],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

train_yami1=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline,
        version=angle_version),
train_yami2=dict(
        type=dataset_type,
        ann_file='/home/zf/Dataset/1yami11k/' + 'annfiles/',
        img_prefix='/home/zf/Dataset/1yami11k/' + 'images/',
        pipeline=train_pipeline,
        version=angle_version),
train_airforce=dict(
        type=dataset_type,
        ann_file='/home/zf/Dataset/2airforce8k/' + 'annfiles/',
        img_prefix='/home/zf/Dataset/2airforce8k/' + 'images/',
        pipeline=train_pipeline,
        version=angle_version),
train_gaofen=dict(
        type=dataset_type,
        ann_file='/home/zf/Dataset/3gaofenship30k/' + 'annfiles/',
        img_prefix='/home/zf/Dataset/3gaofenship30k/' + 'images/',
        pipeline=train_pipeline,
        version=angle_version),
train_fairm=dict(
        type=dataset_type,
        ann_file='/home/zf/Dataset/4fairm_hjj18k/' + 'annfiles/',
        img_prefix='/home/zf/Dataset/4fairm_hjj18k/' + 'images/',
        pipeline=train_pipeline,
        version=angle_version),
train_dota=dict(
        type=dataset_type,
        ann_file='/home/zf/Dataset/5dota5k/' + 'annfiles/',
        img_prefix='/home/zf/Dataset/5dota5k/' + 'images/',
        pipeline=train_pipeline,
        version=angle_version),
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train = [
        # train_yami1,
        # train_yami2,
        # train_airforce,
        train_gaofen,
        # train_fairm,
        # train_dota
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/images/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline,
        version=angle_version))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0002, #/8*gpu_number,
    betas=(0.9, 0.999),
    weight_decay=0.05)
load_from = '/home/zf/Large-Selective-Kernel-Network/checkpoints/lsknet_s_fair_epoch12.pth'
resume_from = None
fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=3, metric='mAP')
checkpoint_config = dict(interval=3)