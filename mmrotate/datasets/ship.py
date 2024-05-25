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
class ShipDataset(CustomDataset):
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
    NAME_LABEL_MAP_Gaofen_ship ={
    'Emory S. Land-class submarine tender':1, 
    'Submarine':2, 
    'Barracks Ship':3, 
    'Whidbey Island-class dock landing ship':4, 
    'San Antonio-class amphibious transport dock':5, 
    'Arleigh Burke-class Destroyer':6, 
    'Ticonderoga-class cruiser':7, 
    'Barge':8, 
    'Sand Carrier':9, 
    'Oliver Hazard Perry-class frigate':10, 
    'Towing vessel':11, 
    'unknown':12, 
    '022-missile boat':13, 
    '037-submarine chaser':14, 
    'Bunker':15, 
    '904B-general stores issue ship':16, 
    '072III-landing ship':17, 
    '926-submarine support ship':18, 
    'Independence-class littoral combat ship':19, 
    'Avenger-class mine countermeasures ship':20, 
    'Mercy-class hospital ship':21, 
    '052D-destroyer':22, 
    '074-landing ship':23, 
    '529-Minesweeper':24, 
    'USNS Bob Hope':25, 
    '051-destroyer':26, 
    'Fishing Vessel':27, 
    'Zumwalt-class destroyer':28, 
    'Wasp-class amphibious assault ship':29, 
    'Freedom-class littoral combat ship':30, 
    'Container Ship':31, 
    'Tuzhong Class Salvage Tug':32, 
    'Other Warship':33, 
    '922A-Salvage lifeboat':34, 
    '903A-replenishment ship':35, 
    'Sovremenny-class destroyer':36, 
    '056-corvette':37, 
    'Bulk carrier':38, 
    'Nimitz Aircraft Carrier':39, 
    '272-icebreaker':40, 
    '909-experimental ship':41, 
    '053H2G-frigate':42, 
    'Lewis and Clark-class dry cargo ship':43, 
    '082II-Minesweeper':44, 
    '054A-frigate':45, 
    '051C-destroyer':46, 
    '052B-destroyer':47, 
    '053H3-frigate':48, 
    '081-Minesweeper':49, 
    'Henry J. Kaiser-class replenishment oiler':50, 
    'Tarawa-class amphibious assault ship':51, 
    '052C-destroyer':52, 
    '073-landing ship':53, 
    '072A-landing ship':54, 
    '037-hospital ship':55, 
    'Forrestal-class Aircraft Carrier':56, 
    'Kitty Hawk class aircraft carrier':57, 
    'Northampton-class tug':58, 
    '815-spy ship':59, 
    '054-frigate':60, 
    'Traffic boat':61, 
    'Hibiki-class ocean surveillance ships':62, 
    'Hatsuyuki-class destroyer':63, 
    'Uraga-class Minesweeper Tender':64, 
    'Hayabusa-class guided-missile patrol boats':65, 
    'Abukuma-class destroyer escort':66, 
    'JS Chihaya':67, 
    'Uwajima-class minesweepers':68, 
    'JMSDF LCU-2001 class utility landing crafts':69, 
    'YW-17 Class Yard Water':70, 
    'Iowa-class battle ship':71, 
    '072II-landing ship':72, 
    '071-amphibious transport dock':73, 
    '074A-landing ship':74, 
    'Hiuchi-class auxiliary multi-purpose support ship':75, 
    '815A-spy ship':76, 
    '636-hydrographic survey ship':77, 
    '639A-Hydroacoustic measuring ship':78, 
    '635-hydrographic Survey Ship':79, 
    'unknown auxiliary ship':80, 
    '904-general stores issue ship':81, 
    '903-replenishment ship':82, 
    '053H1G-frigate':83, 
    '920-hospital ship':84, 
    'Powhatan-class tugboat':85, 
    'Tank ship':86, 
    '055-destroyer':87, 
    '925-Ocean salvage lifeboat':88, 
    'Mashu-class replenishment oilers':89, 
    'Kongo-class destroyer':90, 
    'Asagiri-class Destroyer':91, 
    'Takanami-class destroyer':92, 
    'Xu Xiake barracks ship':93, 
    'Yacht':94, 
    'Hatakaze-class destroyer':95, 
    '648-submarine repair ship':96, 
    'Hatsushima-class minesweeper':97, 
    'YG-203 class yard gasoline oiler':98, 
    'Cyclone-class patrol ship':99, 
    'Lewis B. Puller-class expeditionary mobile base ship':100, 
    '917-lifeboat':101, 
    'Osumi-class landing ship':102, 
    'Towada-class replenishment oilers':103, 
    'Sugashima-class minesweepers':104, 
    'Futami-class hydro-graphic survey ships':105, 
    'JS Kurihama':106, 
    '037II-missile boat':107, 
    'Murasame-class destroyer':108, 
    'Tenryu-class training support ship':109, 
    'Kurobe-class training support ship':110, 
    '051B-destroyer':111, 
    '721-transport boat':112, 
    '891A-training ship':113, 
    '679-training ship':114, 
    'North Transfer 990':115, 
    '625C-Oceanographic Survey Ship':116, 
    'Sacramento-class fast combat support ship':117, 
    '909A-experimental ship':118, 
    'YO-25 class yard oiler':119, 
    'Izumo-class helicopter destroyer':120, 
    '001-aircraft carrier':121, 
    '905-replenishment ship':122, 
    '908-replenishment ship':123, 
    '052-destroyer':124, 
    'USNS Spearhead':125, 
    'Akizuki-class destroyer':126, 
    'Hyuga-class helicopter destroyer':127, 
    'Yaeyama-class minesweeper':128, 
    'USNS Montford Point':129, 
    'JS Suma':130, 
    'Blue Ridge class command ship':131, 
    '901-fast combat support ship':132, 
    '680-training ship':133
}
    CLASSES=[]
    for name, label in NAME_LABEL_MAP_Gaofen_ship.items():
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

        super(ShipDataset, self).__init__(ann_file, pipeline, **kwargs)

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
                        cls_name_splits = bbox_info[8:]
                        class_name=''
                        for i in cls_name_splits:
                            class_name +=  i+' '
                        cls_name = class_name[:-1]
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
            writer.writerow(['ImageID','LabelName','X1','Y1','X2','Y2','X3','Y3','X4','Y4','Conf'])
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
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]+[str(bbox[-1])]
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
