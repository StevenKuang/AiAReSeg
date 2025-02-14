import os
import time
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import wandb

import torch
from torch.utils.data.distributed import DistributedSampler
from itertools import islice

from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
from lib.train.trainers import BaseTrainer


class LTRTrainer(BaseTrainer):
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

        super().__init__(actor, loaders, optimizer, settings, lr_scheduler, cfg)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """
        Do a cycle of training or validation.
        """

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()
        # limit = 5
        avg_epoch_loss = 0
        limit = loader.__len__()
        for i, data in islice(enumerate(loader), limit):
            if self.move_data_to_gpu:
                data = data.to(self.device)
            # debug i
            print(str(i) + '/' + str(loader.__len__()))
            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # Forward pass
            # with torch.autograd.detect_anomaly():

            loss, stats = self.actor(data)

            # print([param.requires_grad for param incnet.parameters()])
            # Backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()

                # for tag, value in self.actor.net.named_parameters():
                #     if value.grad is not None:
                #         print(value.grad.cpu())

                # monitor gradients before clipping
                # gradients = np.array([p.grad.norm().item() for p in self.actor.net.parameters() if p.grad is not None])
                # print(f'Mean gradient norm before clip: {np.mean(gradients)}')


                if self.settings.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)

                # monitor gradients
                # gradients = np.array([p.grad.norm().item() for p in self.actor.net.parameters() if p.grad is not None])
                # print(f'Mean gradient norm: {np.mean(gradients)}')
                # wandb.log({'mean_gradient_norm': np.mean(gradients)})

                self.optimizer.step()

                # print('lr/group0: ', self.lr_scheduler.get_lr()[0])
                # print('lr/group1: ', self.lr_scheduler.get_lr()[1])
                # print('before opt_lr/group0: ', self.optimizer.param_groups[0]['lr'])
                # print('before opt_lr/group1: ', self.optimizer.param_groups[1]['lr'])
                # wandb.log({'lr/group0': self.lr_scheduler.get_lr()[0]})
                # wandb.log({'lr/group1': self.lr_scheduler.get_lr()[1]})
                wandb.log({'lr/group0': self.optimizer.param_groups[0]['lr']})
                wandb.log({'lr/group1': self.optimizer.param_groups[1]['lr']})

                if self.lr_scheduler is not None:
                    if self.settings.scheduler_type == 'cosine':
                        # normal cosine annealing
                        # self.lr_scheduler.step()
                        # cosine annealing with warm restarts
                        # if self.epoch >= self.init_epoch:
                        self.lr_scheduler.step(self.epoch - self.init_epoch + i / limit)
                        # else:
                        #     self.lr_scheduler.step(self.epoch + i / limit)

                # print('after opt_lr/group0: ', self.optimizer.param_groups[0]['lr'])
                # print('after opt_lr/group1: ', self.optimizer.param_groups[1]['lr'])
                # print('after opt_lr/group0 inside: ', self.lr_scheduler.optimizer.param_groups[0]['lr'])
                # print('after opt_lr/group1 inside: ', self.lr_scheduler.optimizer.param_groups[1]['lr'])

            # Update statistics
            batch_size = data['search_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # Print statistics
            self._print_stats(i, loader, batch_size)

            avg_epoch_loss += loss.item()
        print(self.epoch - self.init_epoch)
        wandb.log({'epoch_loss': avg_epoch_loss / limit})


    def train_epoch(self):
        """
        Do one epoch for each loader.
        """

        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if self.settings.local_rank in [-1, 0] and i == loader.__len__() - 1:
            print_str = '[%d: %d]' % (self.epoch, loader.__len__())
            print_str += 'FPS: %.1f, ' % (average_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        # Why are we printing all of the losses that we aren't including?
                        print_str += '%s: %.3f, ' % (name, val.avg)

            print(print_str[:-2])

            log_str = print_str[:-2] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)
        # #####New#####
        # elif self.settings.local_rank not in [-1, 0] and i == loader.__len__():
        #     print_str = '[%d: %d]' % (self.epoch, loader.__len__())
        #     print_str += 'FPS: %.1f, ' % (average_fps)
        #     for name, val in self.stats[loader.name].items():
        #         if (self.settings.print_stats is None or name in self.settings.print_stats):
        #             if hasattr(val, 'avg'):
        #                 print_str += '%s: %.3f, ' % (name, val.avg)
        #
        #     print(print_str[:-2])
        #     log_str = print_str[:-2] + '\n'
        #     with open(self.settings.log_file, 'a') as f:
        #         f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    #print(self.stats[loader.name].keys())
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    # def print_grad_hook(module, grad_input, grad_output):
    #     print(f"Current module: {module}")
    #     print(f"Grad input: {grad_input}")
    #     print(f"Grad output: {grad_output}")