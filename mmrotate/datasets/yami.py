# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS
import csv
import shutil

headstr = """\
   <annotation>
	<folder></folder>
    <gsd>{}</gsd>
	<filename>{}</filename>
	<path>{}</path>
	<source>
		<database>{}</database>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>{}</depth>
	</size>
	<segmented>0</segmented>
    """
objstr = """\
       <object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>{}</difficult>
		<robndbox>
			<cx>{}</cx>
			<cy>{}</cy>
			<w>{}</w>
			<h>{}</h>
			<angle>{}</angle>
		</robndbox>
		<extra>{}</extra>
	</object>
    """
tailstr = '''\
      </annotation>
    '''

@ROTATED_DATASETS.register_module()
class YamiDataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    # CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    #            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #            'basketball-court', 'storage-tank', 'soccer-ball-field',
    #            'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    NAME_LABEL_MAP_YaMi2={
    'F5':1, 
    'P7':2, 
    'F2':3, 
    'W1':4, 
    'S4':5, 
    'T1':6, 
    'C14':7, 
    'B3':8, 
    'A7':9, 
    'A8':10, 
    'C2':11, 
    'P3':12, 
    'F8':13, 
    'C8':14, 
    'W2':15, 
    'S7':16, 
    'C13':17, 
    'T7':18, 
    'L3':19, 
    'Y1':20, 
    'M2':21, 
    'S5':22, 
    'V1':23, 
    'T2':24, 
    'S6':25, 
    'C10':26, 
    'S1':27, 
    'R2':28, 
    'D2':29, 
    'V2':30, 
    'C9':31, 
    'P2':32, 
    'H1':33, 
    'U2':34, 
    'H3':35, 
    'N1':36, 
    'T5':37, 
    'A9':38, 
    'D1':39, 
    'C6':40, 
    'C5':41, 
    'T8':42, 
    'P5':43, 
    'K2':44, 
    'P4':45, 
    'H2':46, 
    'A3':47, 
    'B1':48, 
    'E2':49, 
    'K3':50, 
    'C12':51,
    'C15':52, 
    'L4':53, 
    'S2':54, 
    'R1':55, 
    'W3':56, 
    'T9':57, 
    'C11':58, 
    'M5':59, 
    'E4':60, 
    'R3':61, 
    'F7':62, 
    'U1':63, 
    'C3':64, 
    'K1':65, 
    'M1':66, 
    'A6':67, 
    'F3':68, 
    'E3':69, 
    'C1':70, 
    'B2':71, 
    'T6':72, 
    'P1':73, 
    'K5':74, 
    'K4':75, 
    'A4':76, 
    'L2':77, 
    'C16':78, 
    'S3':79, 
    'C4':80, 
    'A5':81, 
    'I1':82, 
    'A1':83, 
    'E1':84, 
    'P6':85, 
    'F6':86, 
    'C7':87, 
    'M4':88, 
    'F1':89, 
    'T10':90, 
    'T3':91, 
    'L1':92, 
    'Z1':93, 
    'A2':94, 
    'T4':95, 
    'M3':96, 
    'R4':97, 
    'T11':98
    }
    CLASSES=[]
    for name, label in NAME_LABEL_MAP_YaMi2.items():
        CLASSES.append(name)
    # CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    #            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #            'basketball-court', 'storage-tank', 'soccer-ball-field',
    #            'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    # PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
    #            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
    #            (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
    #            (255, 255, 0), (147, 116, 116), (0, 0, 255)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(YamiDataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.jpg')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg' #change
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        difficulty = 0 #int(bbox_info[9])  change
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError
        # self.format_results( results, submission_dir='./merge_result', nproc=4)
        
        return eval_results

    # def merge_det(self, results, nproc=4):
    #     """Merging patch bboxes into full image.

    #     Args:
    #         results (list): Testing results of the dataset.
    #         nproc (int): number of process. Default: 4.
    #     """
    #     collector = defaultdict(list)
    #     for idx in range(len(self)):
    #         result = results[idx]
    #         img_id = self.img_ids[idx]
    #         splitname = img_id.split('__')
    #         oriname = splitname[0]
    #         pattern1 = re.compile(r'__\d+___\d+')
    #         x_y = re.findall(pattern1, img_id)
    #         x_y_2 = re.findall(r'\d+', x_y[0])
    #         x, y = int(x_y_2[0]), int(x_y_2[1])
    #         new_result = []
    #         for i, dets in enumerate(result):
    #             bboxes, scores = dets[:, :-1], dets[:, [-1]]
    #             ori_bboxes = bboxes.copy()
    #             ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
    #                 [x, y], dtype=np.float32)
    #             labels = np.zeros((bboxes.shape[0], 1)) + i
    #             new_result.append(
    #                 np.concatenate([labels, ori_bboxes, scores], axis=1))

    #         new_result = np.concatenate(new_result, axis=0)
    #         collector[oriname].append(new_result)

    #     merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
    #     if nproc <= 1:
    #         print('Single processing')
    #         merged_results = mmcv.track_iter_progress(
    #             (map(merge_func, collector.items()), len(collector)))
    #     else:
    #         print('Multiple processing')
    #         merged_results = mmcv.track_parallel_progress(
    #             merge_func, list(collector.items()), nproc)

    #     return zip(*merged_results)

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            # pattern1 = re.compile(r'__\d+___\d+')
            # x_y = re.findall(pattern1, img_id)
            # x_y_2 = re.findall(r'\d+', x_y[0])
            # x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                # ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                #     [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            # os.rmdir(out_folder)
            shutil.rmtree(out_folder, ignore_errors=True)
            # raise ValueError(f'The out_folder should be a non-exist path, '
            #                  f'but {out_folder} is existing')
        os.makedirs(out_folder)
        filename = out_folder+'.csv'

        #写入csv文件
        with open(filename,'w',newline='') as file:
            writer =csv.writer(file)
            writer.writerow(['ImageID','LabelName','Conf','X1','Y1','X2','Y2','X3','Y3','X4','Y4'])
            for img_id, dets_per_cls in zip(id_list, dets_list):#处理每一张图片，每一张图片有遍历每一个类别
                xml_filename=os.path.join(out_folder,img_id+'.xml')
                head=headstr.format('0.5',img_id,out_folder,'yami',1024,1024,1)
                f = open(xml_filename, "w",encoding='utf-8')
                f.write(head)
                for i , dets in  enumerate(dets_per_cls):#开始处理每一个检测的类别
                    if dets.size == 0:
                        continue
                    bboxes = obb2poly_np(dets, self.version)
                    for bbox in bboxes:
                        csv_element = [img_id+'.tif', self.CLASSES[i]
                                   ] +[str(bbox[-1])]+ [f'{p:.2f}' for p in bbox[:-1]]
                        writer.writerow(csv_element)
                    # if dets.shape[0]==1:
                    #     dets = [dets]
                    for det in dets:
                        if det[5]>0.3: 
                            obj=objstr.format(self.CLASSES[i],0,det[0],det[1],det[2],det[3],det[4],det[5])
                            f.write(obj)
                f.write(tailstr)
                f.close()
                    
        # #写入txt文件
        # files = [
        #     osp.join(out_folder, 'Task1_' + cls + '.txt')
        #     for cls in self.CLASSES
        # ]
        # file_objs = [open(f, 'w') for f in files]
        # for img_id, dets_per_cls in zip(id_list, dets_list):
        #     for f, dets in zip(file_objs, dets_per_cls):
        #         if dets.size == 0:
        #             continue
        #         bboxes = obb2poly_np(dets, self.version)
        #         for bbox in bboxes:
        #             txt_element = [img_id, str(bbox[-1])
        #                            ] + [f'{p:.2f}' for p in bbox[:-1]]
        #             f.writelines(' '.join(txt_element) + '\n')

        # for f in file_objs:
        #     f.close()

        # target_name = osp.split(out_folder)[-1]
        # with zipfile.ZipFile(
        #         osp.join(out_folder, target_name + '.zip'), 'w',
        #         zipfile.ZIP_DEFLATED) as t:
        #     for f in files:
        #         t.write(f, osp.split(f)[-1])

        # return files
        return filename

    # def _results2submission(self, id_list, dets_list, out_folder=None):
    #     """Generate the submission of full images.

    #     Args:
    #         id_list (list): Id of images.
    #         dets_list (list): Detection results of per class.
    #         out_folder (str, optional): Folder of submission.
    #     """
    #     if osp.exists(out_folder):
    #         raise ValueError(f'The out_folder should be a non-exist path, '
    #                          f'but {out_folder} is existing')
    #     os.makedirs(out_folder)

    #     files = [
    #         osp.join(out_folder, 'Task1_' + cls + '.txt')
    #         for cls in self.CLASSES
    #     ]
    #     file_objs = [open(f, 'w') for f in files]
    #     for img_id, dets_per_cls in zip(id_list, dets_list):
    #         for f, dets in zip(file_objs, dets_per_cls):
    #             if dets.size == 0:
    #                 continue
    #             bboxes = obb2poly_np(dets, self.version)
    #             for bbox in bboxes:
    #                 txt_element = [img_id, str(bbox[-1])
    #                                ] + [f'{p:.2f}' for p in bbox[:-1]]
    #                 f.writelines(' '.join(txt_element) + '\n')

    #     for f in file_objs:
    #         f.close()

    #     target_name = osp.split(out_folder)[-1]
    #     with zipfile.ZipFile(
    #             osp.join(out_folder, target_name + '.zip'), 'w',
    #             zipfile.ZIP_DEFLATED) as t:
    #         for f in files:
    #             t.write(f, osp.split(f)[-1])

    #     return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results
