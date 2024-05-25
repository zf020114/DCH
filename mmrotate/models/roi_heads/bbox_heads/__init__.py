# Copyright (c) OpenMMLab. All rights reserved.
from .convfc_rbbox_head import (RotatedConvFCBBoxHead,
                                RotatedKFIoUShared2FCBBoxHead,
                                RotatedShared2FCBBoxHead,
                                RotatedShared2FCBBoxHead_double,
                                RotatedShared2FCBBoxHead_double_filter,
                                RotatedConvFCBBoxHead_arcface,
                                RotatedConvFCBBoxHead_Patitial_FC_double,
                                RotatedShared2FCBBoxHead_double_hard,
                                RotatedShared2FCBBoxHead_treble)
from .gv_bbox_head import GVBBoxHead
from .rotated_bbox_head import RotatedBBoxHead

__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead',
    'GVBBoxHead', 'RotatedKFIoUShared2FCBBoxHead','RotatedShared2FCBBoxHead_double',
    'RotatedShared2FCBBoxHead_double_filter','RotatedConvFCBBoxHead_arcface','RotatedConvFCBBoxHead_Patitial_FC_double',
    'RotatedShared2FCBBoxHead_double_hard','RotatedShared2FCBBoxHead_treble'
]
