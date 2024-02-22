import torch
import functools
import wandb

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import lib.train.losses.flow_reconstruction as flow_reconstruction
from lib.test.utils.visualisation import flow2rgb_torch, get_meshgrid
from lib.train.data import flow_utils
import matplotlib.pyplot as plt

# logger = utils.log.getLogger(__name__)

class ReconstructionLoss:
    def __init__(self, cfg, model):
        self.criterion = nn.MSELoss() if cfg.GWM.CRITERION == 'L2' else nn.L1Loss()
        self.l1_optimize = cfg.GWM.L1_OPTIMIZE
        self.homography = cfg.GWM.HOMOGRAPHY
        self.device = model.device
        self.cfg = cfg
        self.grid_x, self.grid_y = get_meshgrid(cfg.GWM.RESOLUTION, model.device)
        # self.mult_flow = cfg.GWM.USE_MULT_FLOW
        self.flow_colorspace_rec = cfg.GWM.FLOW_COLORSPACE_REC
        flow_reconstruction.set_subsample_skip(cfg.GWM.HOMOGRAPHY_SUBSAMPLE, cfg.GWM.HOMOGRAPHY_SKIP)
        self.flow_u_low = cfg.GWM.FLOW_CLIP_U_LOW
        self.flow_u_high = cfg.GWM.FLOW_CLIP_U_HIGH
        self.flow_v_low = cfg.GWM.FLOW_CLIP_V_LOW
        self.flow_v_high = cfg.GWM.FLOW_CLIP_V_HIGH

        self._recon_fn = self.flow_quad
        # logger.info(f'Using reconstruction method {self._recon_fn.__name__}')
        self.it = 0
        self._extra_losses = []
        self.mse_weight = cfg.TRAIN.MSE_RECONSTRUCTION_WEIGHT
        self.l1_weight = cfg.TRAIN.L1_RECONSTRUCTION_WEIGHT

    def __call__(self, sample, flow, masks_softmaxed, it, train=True):
        return self.loss(sample, flow, masks_softmaxed, it, train=train)

    def loss(self, sample, flow, mask_softmaxed, it, train=True):
        self.training = train
        flow = self.process_flow(sample, flow)

        if self.cfg.GWM.FLOW_RES is not None:
            if flow.shape[-2:] != mask_softmaxed.shape[-2:]:
                print(f'Resizing predicted masks to {self.cfg.GWM.FLOW_RES}')
                mask_softmaxed = F.interpolate(mask_softmaxed, flow.shape[-2:], mode='bilinear', align_corners=False)

        rec_flows = self.rec_flow(sample, flow, mask_softmaxed)

        if self.l1_optimize:
            self._extra_losses.extend([nn.L1Loss()(flow, rec_flow) for rec_flow in rec_flows])

        if not isinstance(rec_flows, (list, tuple)):
            rec_flows = (rec_flows,)

        k = len(rec_flows)
        loss = sum(self.criterion(flow, rec_flow) / k for rec_flow in rec_flows)
        wandb.log({"rec_mse_loss": loss.item()})
        if isinstance(sum(self._extra_losses, 0.), torch.Tensor):
            wandb.log({"rec_l1_loss": sum(self._extra_losses, 0.).item() / len(self._extra_losses)})

        if len(self._extra_losses):
            loss = loss * self.mse_weight + sum(self._extra_losses, 0.) / len(self._extra_losses) * self.l1_weight
        self._extra_losses = []

        wandb.log({"rec_loss": loss.item()})

        # # visualize the rec_flow
        # for i in range(len(rec_flows[0])):
        #     rec_flow = rec_flows[0][i]
        #     mask_bin = mask_softmaxed[i]
        #     flow_orig = flow[i]
        #     rgb_mask = torch.stack([mask_bin, mask_bin, mask_bin], 0).squeeze(1).detach().permute(1, 2,
        #                                                                                           0).cpu().numpy() * 255
        #     rgb_rec_flow = torch.tensor(
        #         flow_utils.flow2img(rec_flow.detach().cpu().permute(1, 2, 0).numpy())).float() / 255.0
        #     rgb_orig_flow = torch.tensor(
        #         flow_utils.flow2img(flow_orig.detach().cpu().permute(1, 2, 0).numpy())).float() / 255.0
        #     fig, ax = plt.subplots(1, 3)
        #     ax[0].imshow(rgb_mask)
        #     ax[1].imshow(rgb_rec_flow)
        #     ax[2].imshow(rgb_orig_flow)
        #     # add caption to fig
        #     # fig.suptitle(f"rec loss:" + str(losses[i].item()))
        #     plt.show()
        #     # wandb.log({"rec_flow_viz": [wandb.Image(fig, caption="rec_flow_viz")]})
        self.it = self.it + 1
        return loss

    def flow_quad(self, sample, flow, masks_softmaxed, it, **_):
        # print(f'Reconstruction using quadratic. Masks shape {masks_softmaxed.shape} | '
        #                   f'Flow shape {flow.shape} | '
        #                   f'Grid shape {self.grid_x.shape, self.grid_y.shape}')
        return flow_reconstruction.get_quad_flow(masks_softmaxed, flow, self.grid_x, self.grid_y)

    def _clipped_recon_fn(self, *args, **kwargs):
        flow = self._recon_fn(*args, **kwargs)
        flow_o = flow[:, :-2]
        flow_u = flow[:, -2:-1].clip(self.flow_u_low, self.flow_u_high)
        flow_v = flow[:, -1:].clip(self.flow_v_low, self.flow_v_high)
        return torch.cat([flow_o, flow_u, flow_v], dim=1)

    def rec_flow(self, sample, flow, masks_softmaxed):
        it = self.it
        if self.cfg.GWM.FLOW_RES is not None and flow.shape[-2:] != self.grid_x.shape[-2:]:
            # print(f'Generating new grid predicted masks of {flow.shape[-2:]}')
            self.grid_x, self.grid_y = get_meshgrid(flow.shape[-2:], self.device)
        return [self._clipped_recon_fn(sample, flow, masks_softmaxed, it)]

    def process_flow(self, sample, flow_cuda):
        return flow_cuda

    def viz_flow(self, flow):
        return torch.stack([flow2rgb_torch(x) for x in flow])

