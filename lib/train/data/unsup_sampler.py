import random

import torch.utils.data

from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils

# For debugging
import matplotlib.pyplot as plt
import numpy as np
from lib.train.dataset import Catheter_tracking
from lib.train.data import opencv_loader
import cv2

def no_processing(data):
    return data


class UnsupervisedSampler(torch.utils.data.Dataset):
    """
    Class responsible for sampling frames from training sequences to form batches.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing.
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap, processing=no_processing, pos_prob=0.5,segmentation=True):
        """
        Args:
            datasets: List of datasets to be used for training.     (We have only 1 here)
            p_datasets: List containing the probabilities by which each dataset will be sampled. [1]
            samples_per_epoch: Number of training samples per epoch.
            max_gap: Maximum gap, in frame numbers, between the train frames and the test frames.
            processing: An instance of Processing class which performs the necessary processing of the data.
        """
        self.segmentation = segmentation
        self.datasets = datasets
        self.pos_prob = pos_prob  # Probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.processing = processing

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """
        Samples num_ids frames between min_id and max_id for which target is visible.

        Args:
            visible: 1D Tensor indicating whether target is visible for each frame.
            min_id: Minimum allowed frame number.
            max_id: Maximum allowed frame number.

        Returns:
            list: List of sampled frame numbers. None if not sufficient visible frames could be found.
        """

        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        # Get valid IDs
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        elif allow_invisible:
            valid_ids = [i for i in range(min_id, max_id)]
        else:
            valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible IDs
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        return self.getitem()

    def getitem(self):
        """
        Returns:
            TensorDict: Dict containing all the data blocks.
        """

        valid = False
        data = None
        num_short_refer = 1
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # Sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                search_frame_ids = None
                reference_frame_ids = None
                gap_add = 0

                # Sample test and train frames in a causal manner, i.e. search_frame_ids > reference_frame_ids
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, min_id=num_short_refer, max_id=len(visible) - 1)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=num_short_refer,
                                                              min_id=base_frame_id[0] - self.max_gap * num_short_refer - gap_add,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_add += 5
                        continue
                    selected_frame_ids = base_frame_id + prev_frame_ids
                    selected_frame_ids.sort()
                    reference_frame_ids = selected_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible,
                            min_id=reference_frame_ids[-1] + 1,
                            max_id=reference_frame_ids[-1] + self.max_gap + gap_add)
                    # Increase gap until a frame is found
                    gap_add += 5
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                search_frame_ids = [1]
                reference_frame_ids = []
                for _ in range(num_short_refer + 1):
                    reference_frame_ids.append(1)

            search_frames, search_next_frames, search_flow, search_flow_bboxes, _ = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict, mode='search')
            reference_frames, reference_flow, reference_flow_bboxes, _ = dataset.get_frames(seq_id, reference_frame_ids, seq_info_dict, mode='reference')

            #TODO: Need a if statement here to take care of the segmentation case

            # After plotting, we see that the search frames and reference frames are ok
            # search_frames = np.array(reference_frames[0])
            # plt.imshow(search_frames)
            # plt.show()

            # unique_labels = torch.unique(search_flow['flow'][0])


            data = TensorDict({'search_images': search_frames,
                                'search_next_image': search_next_frames,
                                'search_flow': search_flow['flow'],
                                'search_flow_bboxes': search_flow_bboxes,
                                'search_images_o': search_frames[0],
                                'reference_images': reference_frames,
                                'reference_flow': reference_flow['flow'],
                                'reference_flow_bboxes': reference_flow_bboxes,
                                'dataset': dataset.get_name()})

            # for s in ['search', 'reference']:
            #     prutils.visualize_flow_bbox(data[s + '_flow_bboxes'], data[s + '_flow'], 'xywh')

            # TODO: Keep an eye out for the processing module
            data = self.processing(data)

            # for s in ['search', 'reference']:
            #     prutils.visualize_flow_bbox(data[s + '_flow_bboxes'], data[s + '_flow'], 'xywh')

            # plt.imshow(data['search_images'][0])
            # plt.show()

            # Check whether data is valid
            valid = data['valid']

        return data

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        while True:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames -- was getting masks
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (2 + 1) and len(visible) >= 20

            if enough_visible_frames or not is_video_dataset:
                return seq_id, visible, seq_info_dict

# if "__main__" == __name__:
#
#     # Load in the dataset
#     # After it is loaded, we want to
#     root = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Final_Dataset_Rot/Images"
#     dataset = Catheter_tracking(root, image_loader=opencv_loader, mode='Train')
#     sampler = TrackingSampler(datasets=[dataset], p_datasets=[1.0], samples_per_epoch=1000, max_gap=10, processing=no_processing, pos_prob=0.5)
#
#     data = sampler.getitem()
#     print(data)