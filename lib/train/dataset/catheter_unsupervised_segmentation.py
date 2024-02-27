import collections
import csv
import os
import os.path
import random
from collections import OrderedDict

import numpy as np
import pandas
import pandas as pd
import torch
import torchvision.transforms as transforms

# import sys
# sys.path.append("/media/liming/Data/IDP/AiAProj/AiAReSeg")

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from lib.train.data import opencv_loader
from lib.train.data import pil_loader
from lib.train.data import flow_utils
import lib.train.data.processing_utils as prutils
from .base_video_dataset import BaseVideoDataset


import cv2
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
from glob import glob
from os.path import join

class Catheter_unsupervised_segmentation(BaseVideoDataset):

    def __init__(self, root, image_loader=opencv_loader, vid_ids=None, mode='Train'):
        """
        Args:
            root: Path to the transverse catheter segmentation = dataset.
            image_loader (jpeg4py_loader): The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                           is used by default.
            vid_ids: List containing the IDs of the videos used for training. Note that the sequence IDs are not always the same, there are different IDs for each of the patient images.
            split: If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                   vid_ids or split option can be used at a time.
            data_fraction: Fraction of dataset to be used. The complete dataset is used by default.
        """
        self.root = env_settings().catheter_transverse_segmentation_dir if root is None else root
        # self.root = os.path.join(self.root, 'Images',mode)      
        self.data_dir = env_settings().catheter_transverse_segmentation_dir
        trim_rule = []
        simu_list = []
        # self.render_size = [-1, -1]
        if mode == 'Train':
            simu_list = ["01", "21", "09", "29"]
            trim_rule = [0, 0, 0, 0]
            # self.simu_list = ["01", "21", "09", "29"]
            # self.trim_rule = [30, 40, 30, 15]
            # simu_list = ["21"]
            # trim_rule = [40]
        elif mode == 'Val':
            simu_list = ["31"] 
            trim_rule = [40]
        else:
            AttributeError("The mode is not recognized")

        roots = []
        self.bboxes_dic = {}
        # roots = [join(self.root, "rotations_transverse_" + simu + '/') for simu in simu_list]
        for i in range(len(simu_list)):
            simu = simu_list[i]
            curr_root = join(self.root, "rotations_transverse_" + simu + '/')
            roots.append(curr_root)
            bbox_file = join(curr_root, "bboxes.pt")
            if os.path.exists(bbox_file):
                self.bboxes_dic[simu] = torch.load(bbox_file).to('cpu')
            else:
                raise FileNotFoundError("The bbox file " + bbox_file + " does not exist")


        self.image_loader = image_loader
        super().__init__('Catheter_transverse_segmentation_dir', self.root, self.image_loader)

        self.class_list = [root.split('/')[-2] for root in roots]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.seq_per_class = self._build_class_list(roots, trim_rule)
        self.sequence_list = self._build_sequence_list(roots, trim_rule)


    # We will keep it simple and the datasplitting will be done manually instead of programatically
    def _build_sequence_list(self, roots, trim_rule):
        # Try to access the directories in the root folder of the dataset and get the sequence list
        return self._get_train_sequences(roots, trim_rule)

    def _get_train_sequences(self, roots, trim_rule):
        # Loop through the subdirectory of the training folder to get all the names of the training sequences
        seq_list = []
        for i in range(len(roots)):
            root = roots[i]
            trim = trim_rule[i]
            image_root = join(root, 'filtered')
            subfolders = sorted(glob(join(image_root, '*')))
            for j in range(len(subfolders)):
                if j < trim:
                    continue
                folder = subfolders[j]
                seq_list.append(join(folder.split('/')[-3], folder.split('/')[-2], folder.split('/')[-1]))

        return seq_list


    def _build_class_list(self, roots, trim_rule):
        seq_per_class = {}
        for i in range(len(roots)):
            root = roots[i]
            trim = trim_rule[i]
            class_name = root.split('/')[-2]
            image_root = join(root, 'filtered')
            subfolders = sorted(glob(join(image_root, '*')))
            seq_per_class[class_name] = [subfolder_num for subfolder_num in (int(subfolder.split('_')[-1]) for subfolder in subfolders) if subfolder_num > trim]

        return seq_per_class

    def get_name(self):
        return 'Catheter_unsupervised_segmentation'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)
    
    def _read_gt_flows(self,seq_path):
        flow_folder_path = seq_path.replace("filtered", "flow_cactuss_flownet2")
        # print("_read_gt_flows path: " + flow_folder_path)
        gt_flows = []
        valid = []
        flow_files = sorted(glob(join(flow_folder_path, '*.flo')))
        # get flow shape from the first flow file
        flow_shape = flow_utils.read_gen(flow_files[0]).shape

        for i in range(len(flow_files)):
            file = flow_files[i]
            bbox = self.bboxes_dic[seq_path.split('/')[-3].split('_')[-1]][
                int(seq_path.split('/')[-1].split('_')[-1]), i]
            if (bbox == 0).all():
                valid.append(False)
                # append an empty tensor of the same shape as the flow
                gt_flows.append(torch.zeros(flow_shape))
            else:
                valid.append(True)
                flow = flow_utils.read_gen(file)
                flow_tensor = torch.tensor(flow)
                gt_flows.append(flow_tensor)

        return torch.stack(gt_flows), torch.tensor(valid)    # this is all flow in a label00x folder

    def _read_target_visible(self,seq_path):
        pass

    def _get_sequence_path(self, seq_id, training_mode="Train"):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):

        seq_path = self._get_sequence_path(seq_id)
        flows, valid = self._read_gt_flows(seq_path)
        # # DEBUG: plot all the valid flows and bbox
        # for i in range (len(flows)):
        #     if valid[i]:
        #         flow = flows[i]
        #         bbox = self.bboxes_dic[seq_path.split('/')[-3].split('_')[-1]][int(seq_path.split('/')[-1].split('_')[-1]), i]
        #         prutils.visualize_flow_bbox([bbox], [flow], 'xywh')
                
        return {'flow': flows, 'valid': valid, 'visible': valid}

    def _get_frame_path(self, seq_path, frame_id):
        return join(seq_path + "/%06d"%(frame_id+0) + '.png')

    def _get_frame(self, seq_path, frame_id):

        img = self.image_loader(self._get_frame_path(seq_path, frame_id))
        #img = cv2.resize(img,(320,320))
        return img

    def _get_class_init(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-3]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class_init(seq_path)

        return obj_class


    def get_frames(self, seq_id, frame_ids, seq_info_dict=None, mode='search'):
        seq_path = self._get_sequence_path(seq_id)
        #input(seq_path)
        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        scan_num_str = obj_class.split('_')[-1]
        seq_num = int(seq_path.split('/')[-1].split('_')[-1])
        search_next_frames = None
        if mode == 'search':
            search_next_frames = [self._get_frame(seq_path, f_id+1) for f_id in frame_ids]

        if seq_info_dict is None:
            seq_info_dict = self.get_sequence_info(seq_id)

        gt_flows = {}
        flows_bboxes = [self.bboxes_dic[scan_num_str][seq_num, f_id] for f_id in frame_ids]
        for key, value in seq_info_dict.items():
            gt_flows[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        # # visualize the flow
        # temp_flow = gt_flows['flow'][0].cpu().numpy()
        # temp_flow = flow_utils.flow2img(temp_flow)      # returns a np.uint8(img) of the flow
        # plt_tensor = torch.tensor(temp_flow).float() / 255.0
        #
        # img = frame_list[0]
        # output = img.astype(float) / 255.0
        # # img = output
        # # output = cv2.addWeighted(output, 1, plt_tensor, 0.5, 0)
        #
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(img)
        # axs[1].imshow(plt_tensor)
        # plt.show()
        # # print('success')


        # prutils.visualize_flow_bbox(flows_bboxes, gt_flows['flow'])

        if mode == 'search':
            return frame_list, search_next_frames, gt_flows, flows_bboxes, object_meta
        elif mode == 'reference':
            return frame_list, gt_flows, flows_bboxes, object_meta
        else:
            raise ValueError("Invalid mode")

    def convert_cxcywh_2_x1y1wh(self, bbox):

        cx = bbox[0]
        cy = bbox[1]
        w = bbox[2]
        h = bbox[3]

        x1 = np.round(cx - 0.5*w).astype(int)
        y1 = np.round(cy - 0.5*h).astype(int)

        return (x1,y1,w,h)

if '__main__' == __name__:
    # root = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Final_Dataset_Rot/Images/Train"
    # seq_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Final_Dataset_Rot/Images/Train/Catheter/Catheter-66"
    root = "/media/liming/Data/IDP/dataset/us_simulation3_cactuss"
    dataset = Catheter_unsupervised_segmentation(root)
    frame_list, anno_frames, object_meta = dataset.get_frames(seq_id=66, frame_ids=[14], anno=None)
    print("success!")

