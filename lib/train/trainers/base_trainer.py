import glob
import os
import traceback
import wandb

import torch
from torch.utils.data.distributed import DistributedSampler

from lib.train.admin import multigpu

import numpy as np

class BaseTrainer:
    """
    Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function.
    """

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, cfg=None):
        """
        Args:
            actor: The actor for training the network.
            loaders: List of dataset loaders, e.g. [train_loader, val_loader].
                     In each epoch, the trainer runs one epoch for each loader.
            optimizer: The optimizer used for training, e.g. Adam.
            settings: Training settings.
            lr_scheduler: Learning rate scheduler.
        """

        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders
        self.cfg = cfg

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}
        self.init_epoch = 0

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() and settings.use_gpu else 'cpu')

        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        """
        Updates the trainer settings. Must be called to update internal settings.
        """

        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            # New function: specify checkpoint dir
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print('checkpoints will be saved to %s' % self._checkpoint_dir)

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    # print("training with multiple GPUs, checkpoints directory doesn't exist")
                    # print('create checkpoints directory ...')
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True):
        """
        Do training for the given number of epochs.

        Args:
            max_epochs: Max number of training epochs,
            load_latest: Bool indicating whether to resume from latest epoch.
            fail_safe: Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                if load_latest:
                    # self.load_checkpoint(checkpoint=self.settings.env.pretrained_networks)
                    self.load_checkpoint()
                    self.init_epoch = self.epoch
                    max_epochs = max_epochs + self.epoch


                    # Here, you may decide to remove some of the weights in the final layer in order to get better training performance via transfer learning

                for epoch in range(self.epoch + 1, max_epochs + 1):
                    self.epoch = epoch

                    self.train_epoch()


                    # print('lr/group0: ', self.lr_scheduler.get_lr()[0])
                    # print('lr/group1: ', self.lr_scheduler.get_lr()[1])
                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        # else:
                        #     self.lr_scheduler.step(epoch - 1)
                    # Only save the last 5 checkpoints
                    #save_every_epoch = getattr(self.settings, 'save_every_epoch', False)
                    save_every_epoch = True
                    if epoch > (max_epochs - 5) or save_every_epoch or epoch % 100 == 0:
                        if self._checkpoint_dir:
                            if self.settings.local_rank in [-1, 0]:
                                self.save_checkpoint()
            except:
                print('training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('traceback for the error')
                    print(traceback.format_exc())
                    print('restarting training from last epoch ...')
                else:
                    raise

        if self.settings.local_rank in [-1, 0]:
            print('finished training')

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self):
        """
        Saves a checkpoint of the network and other variables.
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,  # <class 'int'>
            'actor_type': actor_type,  # <class 'str'>
            'net_type': net_type,  # <class 'str'>
            'net': net.state_dict(),  # <class 'collections.OrderedDict'>
            'net_info': getattr(net, 'info', None),  # <class 'NoneType'>
            'constructor': getattr(net, 'constructor', None),  # <class 'NoneType'>
            'optimizer': self.optimizer.state_dict(),  # <class 'dict'>
            'stats': self.stats,  # <class 'collections.OrderedDict'>
            'settings': self.settings  # <class 'lib.train.admin.settings.Settings'>
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            # print("directory doesn't exist, creating ...")
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure.
        os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        """
        Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
                if self.settings.local_rank in [-1, 0]:
                    print('checkpoint file found')
            else:
                if self.settings.local_rank in [-1, 0]:
                    print('no matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # Checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('ERROR: no checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = torch.load(checkpoint_path)
        print('loading pretrained model from', checkpoint_path)

        # In the case that we are using transfer learning, we will remove some weights at this point:
        # We will try the following
        # TODO: Remove the weights on the final IOU fully connected prediction layer

        # del checkpoint_dict['net']['iou_head.fc.linear.weight']
        # del checkpoint_dict['net']['iou_head.fc.linear.bias']
        # del checkpoint_dict['net']['iou_head.iou_predictor.weight']
        # del checkpoint_dict['net']['iou_head.iou_predictor.bias']

        if not self.cfg.TRAIN.RESUME:
            # TODO: Remove the weights on the decoder's last few layers
            del checkpoint_dict['net']['transformer.decoder.layers.0.linear1.weight']
            del checkpoint_dict['net']['transformer.decoder.layers.0.linear1.bias']
            del checkpoint_dict['net']['transformer.decoder.layers.0.norm1.weight']
            del checkpoint_dict['net']['transformer.decoder.layers.0.norm1.bias']
            del checkpoint_dict['net']['transformer.decoder.layers.0.norm2.weight']
            del checkpoint_dict['net']['transformer.decoder.layers.0.norm2.bias']
            del checkpoint_dict['net']['transformer.decoder.norm.weight']
            del checkpoint_dict['net']['transformer.decoder.norm.bias']

            # Remove all the weights on the segmentation head
            for key in list(checkpoint_dict['net'].keys()):
                if 'mask_head' in key:
                    del checkpoint_dict['net'][key]

        # TODO: Remove the weights on the tl and br layers
        checkpoint_dict['net_type'] = "AIARESEG"
        checkpoint_dict['actor_type'] = "AIARESEGActor"
        assert net_type == checkpoint_dict['net_type'], 'network is not of correct type'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

        # Never load the scheduler, it exists in older checkpoints
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                #weight = net.backbone._modules['0']._modules['body'].conv1.weight
                #weight_transformer = net.transformer._modules['encoder']._modules['layers']._modules['0'].linear1.weight
                net.load_state_dict(checkpoint_dict[key], strict=False)
                #weight = net.backbone._modules['0']._modules['body'].conv1.weight
                #weight_transformer = net.transformer._modules['encoder']._modules['layers']._modules['0'].linear1.weight

                print("loaded weights")
            elif key == 'optimizer':
                # try:
                #     self.optimizer.load_state_dict(checkpoint_dict[key])
                # except:
                param_dicts = [
                    {'params': [p for n, p in net.named_parameters() if 'backbone' not in n and p.requires_grad]},
                    {'params': [p for n, p in net.named_parameters() if 'backbone' in n and p.requires_grad],
                     'lr': self.cfg.TRAIN.LR * self.cfg.TRAIN.BACKBONE_MULTIPLIER}
                ]
                self.optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.TRAIN.LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

                if self.cfg.TRAIN.SCHEDULER.TYPE == 'cosine':

                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                        T_0=self.cfg.TRAIN.SCHEDULER.T0_EPOCH,
                                                                                        T_mult=self.cfg.TRAIN.SCHEDULER.T_MULT,
                                                                                        eta_min=self.cfg.TRAIN.SCHEDULER.MIN_LR)
                    # iter_per_epoch = int(np.floor(self.cfg.DATA.TRAIN.SAMPLE_PER_EPOCH / self.cfg.TRAIN.BATCH_SIZE))
                    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.TRAIN.EPOCH * iter_per_epoch,
                    #                                                           eta_min=self.cfg.TRAIN.SCHEDULER.MIN_LR)
                elif self.cfg.TRAIN.SCHEDULER.TYPE == 'step':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.cfg.TRAIN.LR_DROP_EPOCH)
                else:
                    raise ValueError('Unknown scheduler type: %s' % self.cfg.TRAIN.SCHEDULER.TYPE)
            elif key == 'stats':
                continue
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
            # Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
        return True

    def load_state_dict(self, checkpoint=None):
        # This implementation is most likely wrong
        """
        Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if isinstance(checkpoint, str):
            # Checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('ERROR: no checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        print('loading pretrained model from', checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'network is not of correct type'

        net.load_state_dict(checkpoint_dict['net'], strict=False)

        return True
