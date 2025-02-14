import multiprocessing
import os
import sys
from collections import OrderedDict
from itertools import product

import numpy as np
import torch
import cv2

from lib.test.evaluation import Sequence, Tracker


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict, segmentation: bool):
    """
    Saves the output of the tracker.
    """

    if not os.path.exists(tracker.results_dir):
        print('create tracking result dir', tracker.results_dir)
        os.makedirs(tracker.results_dir)
    if seq.dataset in ['trackingnet', 'got10k']:
        if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
            os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    # Create new folder for these two datasets
    if seq.dataset in ['trackingnet', 'got10k']:
        base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    else:
        base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter=',', fmt='%d')

    def save_mask(file_path, data):
        for idx, frame in enumerate(data):
            # For each frame, we want to save it into an image in the base result path

            segmentation_image = frame[0].unsqueeze(2).repeat(1,1,3).numpy()
            save_path = os.path.join(file_path,f'{idx}.png')
            cv2.imwrite(save_path, segmentation_image)


    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter=',', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter=',', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'target_mask':
            if not os.path.exists(base_results_path):
                print('create tracking result dir', base_results_path)
                os.makedirs(base_results_path, exist_ok=True)       # create parent as well

            save_mask(base_results_path,data)
            # Loop through each of the list in the data and store it inside of a folder


        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        if key == 'time' and seq.dataset in ['trackingnet', 'got10k']:
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8, segmentation=False):
    """
    Runs a tracker on a sequence.
    """

    # Add multiple GPU support
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except Exception as e:
        print('no worker id found')
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {}, Sequence: {}'.format(tracker.name, tracker.parameter_name, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug, segmentation=segmentation)

    else:
        output = tracker.run_sequence(seq, debug=debug, segmentation=segmentation)

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:

        _save_tracker_output(seq, tracker, output, segmentation)


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8, segmentation=False):
    """
    Runs a list of trackers on a dataset.

    Args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """

    multiprocessing.set_start_method('spawn', force=True)

    print('evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, segmentation=segmentation, num_gpu=1)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('done')
