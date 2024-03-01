import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.ops import masks_to_boxes

import lib.train.data.processing_utils as prutils
from lib.utils import TensorDict
from lib.train.data import flow_utils

# For debugging
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm



def read_flo(file):
    with open(file, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            # The .flo file format has 2 channels, one for horizontal (u) and one for vertical (v) flow components
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (H, W, channels)
            return np.resize(data, (h, w, 2))

def flow_to_color(flow_data, max_flow=None):
    # Use OpenCV to convert the flow to a BGR image
    h, w = flow_data.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    flow_data = np.float32(flow_data)
    magnitude, angle = cv2.cartToPolar(flow_data[..., 0], flow_data[..., 1])

    # Normalize the magnitude to fit the HSV color space
    if max_flow is not None:
        magnitude = np.clip(magnitude / max_flow, 0, 1)
    else:
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def count_subfolders(directory):
    # List all entries in the directory
    entries = os.listdir(directory)
    # Count entries that are directories
    subfolder_count = sum(os.path.isdir(os.path.join(directory, entry)) for entry in entries)
    return subfolder_count


def x1y1x2y2_to_x1y1wh(bbox):

    # Convert to the standard format

    x1,y1,x2,y2 = bbox[0,...].tolist()
    w = x2-x1
    h = y2-y1

    return torch.tensor([x1,y1,w,h])

if __name__ == '__main__':
    # The bboxes are stored as a tensor of shape (100, sequence_len, 4), each has (x1, y1, w, h).
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False, help='Path to the dataset', default='/media/liming/Data/IDP/dataset/us_simulation3_cactuss')
    # parser.add_argument('--out', type=str, default=None, help='Path to save the output image. If None, output will not be saved')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root_dir = os.path.join(args.dataset)
    # root_dir = os.path.join('/media/liming/Data/IDP/dataset/us_phantom')
    # out_file = os.path.join(root_dir, "bboxes")
    flow_folder_name = "flow_cactuss_flownet2"
    store = False
    im_shape = (600, 800, 3)

    # loop through the folders in the root directory
    for cath_dir_name in os.listdir(root_dir):
        cath_dir = os.path.join(root_dir, cath_dir_name, flow_folder_name)
        scan_num_str = cath_dir_name.split('_')[-1]
        scan_num = int(scan_num_str)
        out_file_path = os.path.join(root_dir, cath_dir_name, "bboxes.pt")
        if os.path.exists(out_file_path) and store:
            overwrite = input(f"File {out_file_path} already exists. Overwrite? (y/n)")
            if overwrite.lower() != 'y':
                os.remove(out_file_path)
            else:
                continue
        # out_file_path = os.path.join(root_dir, cath_dir_name, "bboxes_" + scan_num_str + ".pt")
        sequence_len = len(os.listdir(os.path.join(cath_dir, os.listdir(cath_dir)[0])))
        # initiate tensor of shape (100, sequence_len, 4) to store the bounding boxes
        bboxes = torch.zeros((100, sequence_len, 4), device=device)
        for seq_dir_name in tqdm(os.listdir(cath_dir)):
            seq_dir = os.path.join(cath_dir, seq_dir_name)
            seq_num = int(seq_dir_name.split('_')[-1])
            for file in os.listdir(seq_dir):
                if file.endswith('.flo'):
                    file_num = int(file.split('_')[-1].split('.')[0])
                    flow = flow_utils.read_gen(os.path.join(seq_dir, file))
                    flow_tensor = torch.tensor(flow)
                    # pad the flow tensor to match the image (600, 800)
                    flow = torch.nn.functional.pad(flow_tensor, (0, 0,
                                                          (im_shape[1] - flow.shape[1]) // 2,
                                                          (im_shape[1] - flow.shape[1]) // 2,
                                                          (im_shape[0] - flow.shape[0]) // 2,
                                                          (im_shape[0] - flow.shape[0]) // 2))
                    flow_tensor.to(device)
                    bounding_box = x1y1x2y2_to_x1y1wh(prutils.flows_to_boxes(flow_tensor.unsqueeze(0)))
                    bboxes[seq_num, file_num] = bounding_box

                    # flow_img = torch.tensor(flow_utils.flow2img(flow_tensor.cpu().numpy())).float() / 255.0
                    # # flow_img = flow_to_color(flow)
                    # plt.imshow(flow_img)
                    # plt.show()
            # plt.close('all')
        # Save the bounding boxes
        if store:
            torch.save(bboxes, out_file_path)
            print(f"Bounding boxes saved to {out_file_path}")


