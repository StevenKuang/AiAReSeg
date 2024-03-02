import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

import lib.train.data.transforms as tfm
from lib.train.data import tracking_sampler, opencv_loader, processing, LTRLoader, unsup_sampler
# Datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, TrackingNet, Catheter_tracking, Catheter_segmentation, Catheter_unsupervised_segmentation


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'search': cfg.DATA.SEARCH.FACTOR,
                                   'reference': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'search': cfg.DATA.SEARCH.SIZE,
                          'reference': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'search': cfg.DATA.SEARCH.CENTER_JITTER,
                                     'reference': cfg.DATA.SEARCH.CENTER_JITTER,
                                     'initial': cfg.DATA.TEMPLATE.CENTER_JITTER}
    settings.scale_jitter_factor = {'search': cfg.DATA.SEARCH.SCALE_JITTER,
                                    'reference': cfg.DATA.SEARCH.SCALE_JITTER,
                                    'initial': cfg.DATA.TEMPLATE.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ['GOT10K_train', 'GOT10K_vot_train', 'LASOT', 'COCO17', 'TRACKINGNET','Catheter_tracking','Catheter_segmentation', 'Catheter_unsupervised_segmentation']
        if name == 'LASOT':
            datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        elif name == 'GOT10K_train':
            datasets.append(Got10k(settings.env.got10k_dir, split='train', image_loader=image_loader))
        elif name == 'GOT10K_vot_train':
            datasets.append(Got10k(settings.env.got10k_dir, split='vot_train', image_loader=image_loader))
        elif name == 'COCO17':
            datasets.append(MSCOCOSeq(settings.env.coco_dir, version='2017', image_loader=image_loader))
        elif name == 'TRACKINGNET':
            datasets.append(
                TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(12)), image_loader=image_loader))
        elif name == 'Catheter_tracking':
            datasets.append(Catheter_tracking(settings.env.catheter_tracking_dir, mode='Train'))
        elif name == 'Catheter_segmentation':
            datasets.append(Catheter_segmentation(settings.env.catheter_segmentation_dir, mode='Train'))
        elif name == 'Catheter_unsupervised_segmentation':
            datasets.append(Catheter_unsupervised_segmentation(settings.env.catheter_transverse_segmentation_dir, mode='Train'))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform: Applies some augmentations
    # We changed the probability of using gray scale to be 0 because the image is gray scale...
    # transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.00),
    #                                 tfm.RandomHorizontalFlip(probability=0.5))
    #
    # transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
    #                                 tfm.RandomHorizontalFlip_Norm(probability=0.5),
    #                                 # tfm.RandomRotation(probability=0.5),
    #                                 # tfm.RandomCropping(probability=0.5),
    #                                 tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.00),
                                    #tfm.RandomHorizontalFlip(probability=0.5)
                                    )

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.RandomRotation(probability=0.3),
                                    #tfm.RandomCropping(probability=0.1),
                                    tfm.Gaussian_Blur(probability=0.3),
                                    tfm.Salt_and_pepper(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
                                    )


    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor
    
    # This data loader adds some noise to the bounding box, then a square is cropped from the image.
    # Please check the processing scipt for more information
    if settings.segmentation:
        if settings.unsupervised:
            data_processing_train = processing.AIATRACKProcessingUnsupSeg(search_area_factor=search_area_factor,
                                                                 output_sz=output_sz,
                                                                 mode='sequence',
                                                                 transform=transform_train,
                                                                 joint_transform=transform_joint,
                                                                 center_jitter_factor=settings.center_jitter_factor,
                                                                 scale_jitter_factor=settings.scale_jitter_factor,
                                                                 settings=settings)
        else:
            data_processing_train = processing.AIATRACKProcessingSeg(search_area_factor=search_area_factor,
                                                                 output_sz=output_sz,
                                                                 mode='sequence',
                                                                 transform=transform_train,
                                                                 joint_transform=transform_joint,
                                                                 center_jitter_factor=settings.center_jitter_factor,
                                                                 scale_jitter_factor=settings.scale_jitter_factor,
                                                                 settings=settings)

    else:
        data_processing_train = processing.AIATRACKProcessing(search_area_factor=search_area_factor,
                                                              output_sz=output_sz,
                                                              center_jitter_factor=settings.center_jitter_factor,
                                                              scale_jitter_factor=settings.scale_jitter_factor,
                                                              mode='sequence',
                                                              transform=transform_train,
                                                              joint_transform=transform_joint,
                                                              settings=settings)

    # Train sampler and loader
    # The sampler is responsible for sampling frames from training sequences to form batches
    # This could be one of the places where things went wrong
    
    if settings.unsupervised:
        dataset_train = unsup_sampler.UnsupervisedSampler(
            datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
            processing=data_processing_train,
            segmentation=settings.segmentation)
    else:
        dataset_train = tracking_sampler.TrackingSampler(
            datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
            processing=data_processing_train,
            segmentation=settings.segmentation)

    # The sampler becomes zero if we are not using distributed processing
    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None

    shuffle = False if settings.local_rank != -1 else True

    # change num_workers to 1 in the cfg file to debug
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    return loader_train


def get_optimizer_scheduler(net, cfg):
    param_dicts = [
        {'params': [p for n, p in net.named_parameters() if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in net.named_parameters() if 'backbone' in n and p.requires_grad],
         'lr': cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER}
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if cfg.TRAIN.SCHEDULER.TYPE == 'cosine':
        iter_per_epoch = int(np.floor(cfg.DATA.TRAIN.SAMPLE_PER_EPOCH / cfg.TRAIN.BATCH_SIZE))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=iter_per_epoch * cfg.TRAIN.SCHEDULER.T0_EPOCH,
                                                                            T_mult=cfg.TRAIN.SCHEDULER.T_MULT, eta_min=cfg.TRAIN.SCHEDULER.MIN_LR)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.EPOCH * iter_per_epoch,
        #                                                           eta_min=cfg.TRAIN.SCHEDULER.MIN_LR)
    elif cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    else:
        raise ValueError('Unknown scheduler type: %s' % cfg.TRAIN.SCHEDULER.TYPE)

    return optimizer, lr_scheduler
