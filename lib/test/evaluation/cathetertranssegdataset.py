import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import torch
import cv2
from os.path import join
from glob import glob
from lib.train.data import flow_utils

class CatheterTransSegDataset(BaseDataset):
    """
    The catheter tracking dataset consists of 50 training sequences and an additional 15 testing sequences.

    All of them are simulated using the ray tracing algorithm from Imfusion Inc.

    """

    # Constructor

    def __init__(self, subset='Val'):
        super().__init__()
        self.base_path = self.env_settings.cathetertransseg_path
        print("Catheter Base Path:")
        print(self.base_path)
        self.simu_list, self.trim_rule = [], []
        if subset == 'Train':
            self.simu_list = ["01", "21", "09", "29"]
            self.trim_rule = [30, 40, 30, 15]
        elif subset == 'Val':
            self.simu_list = ["31"]
            self.trim_rule = [44]
            # self.simu_list = ["21"]
            # self.trim_rule = [0]
        else:
            AttributeError("The mode is not recognized")

        roots = []
        self.bboxes_dic = {}
        for i in range(len(self.simu_list)):
            simu = self.simu_list[i]
            curr_root = join(self.base_path, "rotations_transverse_" + simu + '/')
            roots.append(curr_root)
            bbox_file = join(curr_root, "bboxes.pt")
            if os.path.exists(bbox_file):
                self.bboxes_dic[simu] = torch.load(bbox_file).to('cpu')
            else:
                raise FileNotFoundError("The bbox file " + bbox_file + " does not exist")

        self.base_path = roots[0]
        self.data_path = self.env_settings.cathetertransseg_path

        self.sequence_list = self._get_sequence_list(subset, roots, self.trim_rule)
        self.height = 320
        self.width = 320
        self._set_height_width()
        self.clean_list = self.clean_seq_list()     # not used anywhere


    # A clean sequence list method that grabs the class of each sequence, in our case there is only one class

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('_')[-1]
            clean_lst.append(cls)

        return clean_lst

    # A get sequence list method at runs the construct sequence method

    def sort_seq_names(self,name):
        parts = name.split("-")
        return int(parts[1])

    def _get_sequence_list(self,subset, roots, trim_rule):
        if subset == 'Val':
            # We grab all of the sequences in the folder by doing a walk
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
                    if (self.bboxes_dic[folder.split('/')[-3].split('_')[-1]][j] == 0).all():
                        continue
                    seq_list.append(join(folder.split('/')[-3], folder.split('/')[-2], folder.split('/')[-1]))

            return seq_list



    def get_sequence_list(self, subset='Val'):
        # seq_list = []
        # for s in self.sequence_list:
        #     cons_out = self._construct_sequence(s)
        #     if cons_out.init_info() is not None:
        #         seq_list.append(cons_out)
        # return SequenceList(seq_list)
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    # def _read_bb_anno(self, seq_path):
    #     # For each of the folders of sequences, you need to generate the annotations and put them into an annotation file
    #     seq = seq_path.split('/')[-1]
    #     bb_anno_file = os.path.join(seq_path, f'gt_{seq}.txt')
    #     gt = []
    #
    #     with open(bb_anno_file) as f:
    #         for line in f.readlines():
    #             line = line[2:-3]
    #             line = line.split(",")
    #             line = [float(i) for i in line]
    #             print(line)
    #             gt.append(line)
    #
    #
    #     return np.array(gt)

    def _read_mask_anno(self,seq_path):
        # Here is the segmentation masks, load them and then put it into the same tensor
        # This may be too big, if it is then try to reduce the number of images used
        seq = seq_path.split('/')[-1]
        mask_path = seq_path.replace("filtered", "filtered_masks")

        # Now we start to load the segmentation masks
        gt = []
        filenames = os.listdir(mask_path)
        filenames = sorted(filenames)
        # filenames_index = int(np.floor(0.9*len(filenames)))
        # filenames = filenames[filenames_index:]
        for filename in filenames:
            if filename.endswith(".png"):
                path = os.path.join(mask_path, filename)
                mask = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                mask_tensor = torch.tensor(mask)
                # Convert the image such that we will only have the label 1 for the catheter, or else it makes it zero
                mask_tensor = torch.where(mask_tensor==2, 1, 0).float()
                sum_check = mask_tensor.sum()
                #mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0),(320,320))
                gt.append(mask_tensor)

        return torch.stack(gt)

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


    def _get_sequence_path(self,seq_id,training_mode="Val"):
        seq_name = self.sequence_list[seq_id-self.trim_rule[0]].split('/')[-1]
        return os.path.join(self.base_path, "filtered", seq_name)

    # A construct sequence method that: Grabs the class name, the ground truth annotations, whether the object is occluded, whether the object is out of view, whether the target is visible, and finally arranges the frames list in the form of a sequence.

    def _get_seq_len(self, seq_id):
        path = self._get_sequence_path(seq_id,training_mode="Val")

        #print(path)
        for seq_paths, seq_subdirs, seq_names in os.walk(os.path.join(path)):
            #print(seq_names)
            seq_names = seq_names

        # Now we can grab the length of that list
        return len(seq_names)

    def _construct_sequence(self,sequence_name):

        # This gives you the class name, which for now is just catheter
        class_name = sequence_name.split('/')[0]

        # This will give you the sequence number, which will range from 50-65
        sequence_number = int(sequence_name.split('_')[-1])

        # Join the base path, with the class, with the sequence name, and then finally the ground truth file name
        # anno_path = os.path.join(self.base_path, class_name, sequence_name, f'gt_{sequence_name}.txt')

        seq_path = os.path.join(self.data_path, sequence_name)

        #frames_path = os.path.join(self.base_path, class_name, sequence_name, 'img')
        frames_path = os.path.join(self.data_path, sequence_name)

        # find the first valid bbox
        init_frame_id = None
        init_bbox = None
        for i in range(len(self.bboxes_dic[class_name.split('_')[-1]][sequence_number])):
            if self.bboxes_dic[class_name.split('_')[-1]][sequence_number][i].sum() != 0:
                init_frame_id = i
                init_bbox = self.bboxes_dic[class_name.split('_')[-1]][sequence_number][i]
                break
        if init_frame_id is None:
            raise ValueError("No valid bbox found in the sequence")

        gt_flow_path = join(seq_path.replace("filtered", "flow_cactuss_flownet2"), f"{init_frame_id:06d}.flo")
        gt_flow = torch.tensor(flow_utils.read_gen(gt_flow_path))
        threshold = 0.2
        # create a binary mask from flow
        mask_u = gt_flow.abs()[:, :, 0] > threshold
        mask_v = gt_flow.abs()[:, :, 1] > threshold
        ground_truth_mask_from_flow = (mask_u | mask_v).float().unsqueeze(0)
        ground_truth_mask = {init_frame_id: dict()}
        # pad the mask to the same size as the image
        if ground_truth_mask_from_flow.shape != torch.Size((1, self.height, self.width)):
            ground_truth_mask_from_flow = torch.nn.functional.interpolate(ground_truth_mask_from_flow.unsqueeze(0), (self.height, self.width))
            ground_truth_mask_from_flow = ground_truth_mask_from_flow.squeeze(0)
        ground_truth_mask[init_frame_id]['mask'] = ground_truth_mask_from_flow.squeeze(0)
        ground_truth_mask[init_frame_id]['bbox'] = init_bbox
        #ground_truth_rect = self._read_bb_anno(seq_path)
        # ground_truth_mask = self._read_mask_anno(seq_path)
        #ground_truth_rect = None
        seq_len = self._get_seq_len(sequence_number)

        print("Folder: " + frames_path + "\nInit frame id: " + str(init_frame_id))
        # The number of zeros will depend on the number of frames in the folder

        full_occlusion = np.zeros(seq_len)

        out_of_view = np.zeros(seq_len)

        # We now want to grab the path of each of the frames

        # Easiest way is to do a walk across the files again

        frames_list = []
        for names, subdires, files in os.walk(os.path.join(frames_path)):
            # print(names)
            # print(subdires)
            # print(files)
            for file in files:
                frames_list.append(os.path.join(frames_path, file))

        frames_list = sorted(frames_list)
        target_class = class_name

        #return Sequence(sequence_name, frames_list, 'catheter_tracking', ground_truth_rect.view(-1, 4), object_class=target_class, target_visible=full_occlusion)
        return Sequence(sequence_name, frames_list, 'catheter', None, ground_truth_seg=ground_truth_mask, object_class=target_class, target_visible=full_occlusion)
        # return Sequence(sequence_name, frames_list, 'catheter', ground_truth_rect,
        #                 object_class=target_class, target_visible=full_occlusion)

    def __len__(self):
        return len(self.sequence_list)

    def _set_height_width(self):
        seq_path_0 = os.path.join(self.data_path, self.sequence_list[0])
        for filename in os.listdir(seq_path_0):
            if filename.endswith(".png"):
                img = cv2.imread(os.path.join(seq_path_0, filename))
                self.height, self.width = img.shape[:2]
                return





# if "__main__" == __name__:
#
#     dataset = CatheterDataset(subset='Val')
#     sequence = dataset._construct_sequence(sequence_name='Catheter-690')
#     seq_len = dataset._get_seq_len(seq_id=690)
#     print(seq_len)


