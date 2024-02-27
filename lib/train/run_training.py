import argparse
import importlib
import os
import random
import warnings
import wandb
import random

warnings.filterwarnings('ignore')

import cv2 as cv
import numpy as np
import torch.backends.cudnn
import torch.distributed as dist

torch.backends.cudnn.benchmark = False

# import _init_paths
import lib.train.admin.settings as ws_settings

# profiling tools
from ddtrace.profiling import Profiler



def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None, base_seed=None, segmentation=False, unsupervised=False):
    """
    Run the train script.

    Args:
        script_name: Name of experiment in the 'experiments/' folder.
        config_name: Name of the yaml file in the 'experiments/<script_name>'.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    if save_dir is None:
        print('save_dir dir is not given, use the default dir instead')
    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    # Set seed for different process
    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = 'train/{}/{}'.format(script_name, config_name)
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    settings.cfg_file = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, config_name))
    settings.unsupervised = unsupervised
    settings.segmentation = segmentation
    if settings.unsupervised is True:
        settings.segmentation = True
    expr_module = importlib.import_module('lib.train.train_script')
    expr_func = getattr(expr_module, 'run')

    if settings.local_rank in [-1, 0]:
        print('script_name: {}.py  config_name: {}.yaml'.format(script_name, config_name))

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='run a train scripts in train_settings')
    parser.add_argument('--script', type=str, default='aiatrack', help='name of the train script')
    parser.add_argument('--config', type=str, default='catheter', help='name of the config file')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='set cudnn benchmark on (1) or off (0) (default is on)')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    parser.add_argument('--save_dir', type=str, default='.', help='the directory to save checkpoints and logs')
    parser.add_argument('--seed', type=int, default=32, help='seed for random numbers')
    parser.add_argument('--segmentation', type=bool, default=False, help='Turning on segmentation')
    parser.add_argument('--unsupervised', type=bool, default=False, help='Turning on unsupervised learning using optical flow.')
    args = parser.parse_args()
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)

    wandb.init(
        project="Unsupervised AiAReSeg",
        config={
            # "learning_rate": 0.02,
            "architecture": "CNN+ViT",
            "dataset": "CACTUSS transverse",
            "epochs": 50,
        # }, mode="online"
        }, mode="disabled"
    )

    # Run training line
    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark,
                 local_rank=args.local_rank, save_dir=args.save_dir, base_seed=args.seed, segmentation=args.segmentation, unsupervised=args.unsupervised)


if __name__ == '__main__':
    # prof = Profiler(
    #     env="prod",  # if not specified, falls back to environment variable DD_ENV
    #     service="Unsup AiAReSeg",  # if not specified, falls back to environment variable DD_SERVICE
    #     version="1.0.1",  # if not specified, falls back to environment variable DD_VERSION
    # )
    # prof.start()  # Should be as early as possible, eg before other imports, to ensure everything is profiled

    main()
