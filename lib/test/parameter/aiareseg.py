import os

from lib.config.aiareseg.config import cfg, update_config_from_file
from lib.test.evaluation.environment import env_settings
from lib.test.utils import TrackerParams


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # Update default config from yaml file
    #yaml_file = os.path.join(prj_dir, 'experiments/aiatrack/%s.yaml' % yaml_name)
    # get working directory
    work_dir = os.getcwd()
    # /home/liming/Documents/AiAProj/AiAReSeg/experiments/aiareseg/AiASeg_s+p.yaml
    yaml_file = os.path.join(work_dir, 'experiments/aiareseg/AiASeg_s+p.yaml')
    # yaml_file = "/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/AiAReSeg/experiments/aiareseg/AiASeg_s+p.yaml"
    update_config_from_file(yaml_file)
    params.cfg = cfg

    # Search region
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    #params.checkpoint = os.path.join(save_dir, 'checkpoints/train/aiatrack/%s/AIATRACK_ep%04d.pth.tar' %
    #                                 (yaml_name, cfg.TEST.EPOCH))
    # params.checkpoint = "/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/AiAReSeg/lib/train/checkpoints/train/aiareseg/AiASeg_s+p/AIARESEG_ep0024.pth.tar"
    # /home/liming/Documents/AiAProj/AiAReSeg/pretrained_networks/AIATRACK_ep0500.pth.tar
    params.checkpoint = os.path.join(work_dir, 'pretrained_networks/AIATRACK_ep0500.pth.tar')
    # Whether to save boxes from all queries
    params.save_all_boxes = False
    params.save_all_masks = True

    return params
