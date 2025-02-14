import torch
import wandb

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.merge import merge_feature_sequence
from lib.utils.misc import NestedTensor
from . import BaseActor
import numpy as np
import cv2

# For debugging:
import matplotlib.pyplot as plt
from lib.train.data import flow_utils

import lib.train.losses as losses
class AIARESEGActor(BaseActor):
    """
    Actor for training.
    """
    # def forward_hook(self, module, input, output):
    #     if isinstance(input, tuple):
    #         for i in input:
    #             print(f"{module.__class__.__name__} input size: {i.shape}")
    #     else:
    #         print(f"{module.__class__.__name__} input size: {input.shape}")
    #
    #     if isinstance(output, tuple):
    #         for i in output:
    #             print(f"{module.__class__.__name__} output size: {i.shape}")
    #     else:
    #         print(f"{module.__class__.__name__} output size: {output.shape}")
    #     # print(f"{module.__class__.__name__} input: {input}")
    #     # print(f"{module.__class__.__name__} output: {output}")

    # def backward_hook(self, module, grad_input, grad_output):
    #     print(f"{module.__class__.__name__} grad input: {grad_input.shape}")
    #     print(f"{module.__class__.__name__} grad output: {grad_output.shape}")

    def __init__(self, net, objective, loss_weight, settings, cfg = None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # Batch size
        self.cfg = cfg
        self.device = torch.device('cuda:0')
        if 'reconstruction' in self.objective:
            self.objective['reconstruction'] = losses.ReconstructionLoss(self.cfg, self)

        # self.mask_threshold = torch.nn.Parameter(torch.tensor(0.2).to(self.device), requires_grad=True)
        # Register a hook for each layer
        # for name, layer in self.net.named_children():
        #     layer.__name__ = name
        #     # layer.register_forward_hook(self.forward_hook)
        #     layer.register_full_backward_hook(self.backward_hook)



    def __call__(self, data):
        """
        Args:
            data: The input data, should contain the fields 'reference', 'search', 'gt_bbox'.
            reference_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)

        Returns:
            loss: The training loss.
            status: Dict containing detailed losses.
        """

        # Forward pass
        out_dict = self.forward_pass(data)

        # Compute losses
        # Process the groundtruth
        if self.settings.segmentation:
            if self.settings.unsupervised:
                loss, status = self.compute_losses_flow_seg(out_dict, data, return_status=True)
                wandb.log(status)
                return loss, status
            else:
                gt_anno = data['search_anno'].squeeze(2).squeeze(2).permute(1,0,2,3)
                loss, status = self.compute_losses_seg(out_dict, gt_anno, return_status=True)
                return loss, status
        else:
            gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)
            loss, status = self.compute_losses(out_dict, gt_bboxes[0], data['proposal_iou'])
            return loss, status



    def forward_pass(self, data):

        # We would need to generalize this into the segmentation domain as well!
        # This would mean redesigning the architecture to take into account the

        # Process the search regions (t-th frame)
        search_dict_list = []
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)


        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        if self.settings.segmentation: # If doing segmentation then freeze the weights
            # with torch.no_grad():
            search_back = self.net(img=NestedTensor(search_img, search_att), mode='backbone')
            # Grab the first 4 elements, these are needed for transformer
            # The latter 4 elements are needed for the segmentation decoder
            search_back_short = {k: search_back[k] for i,k in enumerate(search_back) if i < 4}
            search_dict_list.append(search_back_short)
            search_dict = merge_feature_sequence(search_dict_list)

            #########Plotting attention maps for debugging##########
            # # We plot the original, and then all of the attention maps in subsequent layers
            #
            # # Denorm
            # search_img_copy = search_img.permute(0, 2, 3, 1).detach().cpu()  # (b,320,320,3)
            # mean = torch.tensor([0.485, 0.465, 0.406])
            # std = torch.tensor([0.229, 0.224, 0.225])
            # search_img_denorm = (search_img_copy * std) + mean
            #
            # copy_src = search_dict['feat']
            # copy_src = copy_src.permute(1, 2, 0).view(-1, 256, 20, 20)
            # # gt_flow = data['search_flow'].squeeze(0).permute(0, 3, 1, 2)
            #
            # for i in range(copy_src.shape[0]):
            #     plot_img = copy_src[i, 0,...].detach().cpu().numpy().astype(float)
            #     min_val = np.min(plot_img)
            #     max_val = np.max(plot_img)
            #
            #     new_min = 0.0
            #     new_max = 1.0
            #
            #     plot_img_transformed = new_min + ((plot_img - min_val) * (new_max - new_min)) / (max_val - min_val)
            #     search_img_plot = search_img_denorm[i, ...].numpy()
            #     # rgb_flow = torch.tensor(flow_utils.flow2img(gt_flow[i].permute(1,2,0).cpu().numpy())).float() / 255.0
            #
            #     fig, ax = plt.subplots(1, 2)
            #     ax[0].imshow(search_img_plot)
            #     # ax[0].set_title('Cropped denormalized search image')
            #     ax[0].set_title('Cropped search image')
            #     # plt.show()
            #
            #     ax[1].imshow(plot_img_transformed)
            #     ax[1].set_title('Attention map')
            #
            #     # ax[2].imshow(rgb_flow)
            #     # ax[2].set_title('gt flow')
            #     plt.show()

            #########Plotting for debugging##########

            # Process the reference frames
            feat_dict_list = []
            refer_reg_list = []
            temporal_dict = {}
            for i in range(data['reference_images'].shape[0]):
                reference_dict_list = []
                reference_img_i = data['reference_images'][i].view(-1, *data['reference_images'].shape[
                                                                        2:])  # (batch, 3, 320, 320)
                reference_att_i = data['reference_att'][i].view(-1, *data['reference_att'].shape[2:])  # (batch, 320, 320)
                # temp = NestedTensor(reference_img_i, reference_att_i)
                # input(temp.mask.shape)
                output = self.net(img=NestedTensor(reference_img_i, reference_att_i), mode='backbone')
                temporal_dict[f'{i}'] = output
                ref_back_short = {k: output[k] for i, k in enumerate(output) if i < 4}
                reference_dict_list.append(ref_back_short)
                feat_dict_list.append(merge_feature_sequence(reference_dict_list))
                refer_reg_list.append(data['reference_region'][i])

            # Run the transformer and compute losses
            out_embed, _, _, _ = self.net(search_dic=search_dict, refer_dic_list=feat_dict_list,
                                          refer_reg_list=refer_reg_list, mode='transformer')
        else:
            search_back = self.net(img=NestedTensor(search_img, search_att), mode='backbone')
            search_back_short = {k: search_back[k] for i, k in enumerate(search_back) if i < 4}
            search_dict_list.append(search_back_short)
            search_dict = merge_feature_sequence(search_dict_list)

            # Process the reference frames
            feat_dict_list = []
            refer_reg_list = []
            for i in range(data['reference_images'].shape[0]):
                reference_dict_list = []
                reference_img_i = data['reference_images'][i].view(-1, *data['reference_images'].shape[
                                                                        2:])  # (batch, 3, 320, 320)
                reference_att_i = data['reference_att'][i].view(-1,
                                                                *data['reference_att'].shape[2:])  # (batch, 320, 320)
                # temp = NestedTensor(reference_img_i, reference_att_i)
                # input(temp.mask.shape)
                output = self.net(img=NestedTensor(reference_img_i, reference_att_i), mode='backbone')

                reference_dict_list.append(output)
                feat_dict_list.append(merge_feature_sequence(reference_dict_list))
                refer_reg_list.append(data['reference_region'][i])

            # Run the transformer and compute losses
            out_embed, _, _, _ = self.net(search_dic=search_dict, refer_dic_list=feat_dict_list,
                                          refer_reg_list=refer_reg_list, mode='transformer')


        if self.settings.segmentation:

            #out_seg = self.net(out_embed=out_embed, mode='segmentation',seg20=search_back['Channel20'], seg40=search_back['Channel40'], seg80=search_back['Channel80'], seg160=search_back['Channel160'], pos_emb=search_dict['pos'])
            out_seg = self.net(out_embed=out_embed, mode='segmentation', search_outputs=search_back, reference_outputs=temporal_dict)

            # threshold = 0.5
            # out_seg = out_seg > threshold
            # out_seg = out_seg.float()
            # out_seg = torch.tensor(out_seg,requires_grad=True)
            #
            # B,_, _, _ = out_seg.shape
            #
            # fig, axs = plt.subplots(2, B, figsize=(50, 50))
            #
            # for sample in range(B):
            #
            #     debug_mask = out_seg[sample, ...].squeeze(0).detach().cpu().numpy()
            #     debug_img = data['search_images_o'][0,sample,...].detach().cpu().permute(1,2,0).numpy()
            #     # gt_mask = data['search_anno'][0,sample,...].squeeze(0).detach().cpu().permute(1,2,0).repeat(1,1,3)
            #     # gt_mask = torch.where(gt_mask==1, torch.tensor([0.,1.,0.]), torch.tensor([0.,0.,0.])).numpy().astype(float)
            #
            #
            #     mask_normalized = debug_mask.astype(float)
            #     mask_normalized = np.where(mask_normalized>0.5, 1.0, 0.0)
            #     mask_normalized = np.repeat(mask_normalized[:,:,np.newaxis],3,axis=2)
            #     mask_normalized = np.where(mask_normalized==1., [1.,0.,0.],[0.,0.,0.])
            #
            #     img_normalized = debug_img.astype(float)/255.0
            #
            #     mixed_output = cv2.addWeighted(img_normalized, 1, mask_normalized, 0.5, 0)
            #     # mixed_output = cv2.addWeighted(mixed_output, 1, gt_mask, 0.5, 0)
            #
            #     axs[0, sample].imshow(mask_normalized)
            #     axs[0, sample].title.set_text(f'Model Mask frame {sample}')
            #     axs[1, sample].imshow(mixed_output)
            #     axs[1, sample].title.set_text(f'Image frame {sample}')
            #     # axs[2, sample].imshow(gt_mask)
            #     # axs[2, sample].title.set_text(f'Gt frame {sample}')
            #
            # plt.show()

            # print valid gradients in debug watch
            # np.array([p.grad.norm().item() for p in self.net.parameters() if p.grad is not None])
            return out_seg

        # Forward the corner head
        else:
            out_dict = self.net(out_embed=out_embed, proposals=data['search_proposals'],
                                mode='heads')  # out_dict: (B, N, C), outputs_coord: (1, B, N, C)

            return out_dict

    def compute_losses(self, pred_dict, gt_bbox, iou_gt, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError('ERROR: network outputs is NaN! stop training')
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # Compute GIoU and IoU
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # Compute L1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        iou_pred = pred_dict['pred_iou']
        iou_loss = self.objective['iou'](iou_pred, iou_gt)


        # Weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'iou'] * iou_loss
        if return_status:
            # Status for log
            mean_iou = iou.detach().mean()
            status = {'Ls/total': loss.item(),
                      'Ls/giou': giou_loss.item(),
                      'Ls/l1': l1_loss.item(),
                      'Ls/iou': iou_loss.item(),
                      'IoU': mean_iou.item()}
            return loss, status
        else:
            return loss

    def compute_losses_seg(self,out_seg, gt_mask, return_status=True):

        bce_loss = self.objective['BCE'](out_seg,gt_mask)
        try:
            IOU_loss = self.objective['mask_iou'](out_seg, gt_mask)
        except:
            IOU_loss = torch.tensor(0.0).cuda()

        mse = self.objective['MSE'](out_seg,gt_mask)

        # Weighted sum
        loss = self.loss_weight['BCE'] * bce_loss + self.loss_weight['mask_iou'] * IOU_loss + self.loss_weight['MSE'] * mse

        if return_status:
            # Status for log
            status = {'Ls/total': loss.item(),
                      'Ls/bce': bce_loss.item(),
                      'Ls/iou': IOU_loss.item(),
                      'Ls/mse': mse.item()}
            return loss, status
        else:
            return loss

    def compute_losses_flow_seg(self,out_seg, data, return_status=True):
        """optical flow reconstruction loss from the guess what moves model"""

        flow = data['search_flow'][0]
        threshold = 0.2
        # create a binary mask from flow
        mask_u = flow.abs()[:, :, :, 0] > threshold
        mask_v = flow.abs()[:, :, :, 1] > threshold
        mask_from_flow = (mask_u | mask_v).float().unsqueeze(1)
        # # smooth the edges of the mask
        # mask_from_flow = self.smooth_mask_batch(mask_from_flow, sigmaX=4)

        # # debug
        # # visualize the binary mask for debugging, if true then white else black
        # ori_img = data['search_images'][0]
        # # denorm ori_img
        # ori_img = (ori_img * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        # rgb_mask = torch.cat([mask_from_flow.squeeze(0), mask_from_flow.squeeze(0), mask_from_flow.squeeze(0)], dim=1).permute(0, 2, 3, 1).cpu().numpy() * 255
        # rgb_flow = torch.tensor(flow_utils.flow2img(flow[0].cpu().numpy())).float() / 255.0
        # fig, axs = plt.subplots(1, 3)
        # overlay = cv2.addWeighted(ori_img[0].permute(1, 2, 0).cpu().numpy(), 1, rgb_mask[0], 0.2, 0)
        # axs[0].imshow(rgb_mask[0])
        # axs[1].imshow(rgb_flow)
        # axs[2].imshow(overlay)
        # plt.show()

        if self.cfg.TRAIN.USE_RECONSTRUCTION <= 1:
            loss, status = self.compute_losses_seg(out_seg, mask_from_flow, return_status)
            if self.cfg.TRAIN.USE_RECONSTRUCTION < 1:
                return loss, status

        # visualized gt_flow is a tensor of shape (1, B, 2, H, W)
        # gt_flow and out_seg_softmax must be of the same shape
        gt_flow = data['search_flow'].squeeze(0).permute(0, 3, 1, 2)
        # # debug visualisation
        # for flow in gt_flow.squeeze(0):
        #     rgb_flow = torch.tensor(flow_utils.flow2img(flow.cpu().numpy())).float() / 255.0
        #     plt.imshow(rgb_flow)
        #     plt.show()

        # criterions = {'reconstruction': (losses.ReconstructionLoss(self.cfg, self), self.cfg.GWM.LOSS_MULT.REC, lambda x: 1)}
        # criterion = losses.CriterionDict(criterions)
        sample = None
        iteration = 0

        bg_seg = 1 - out_seg
        patched_seg = torch.cat([bg_seg, out_seg], dim=1)
        patched_seg_softmax = torch.nn.functional.softmax(patched_seg, dim=1)

        rec_loss = self.objective['reconstruction'](sample, gt_flow, patched_seg, iteration, train=True)

        if self.cfg.TRAIN.USE_RECONSTRUCTION > 1:
            if return_status:
                status = {
                    'Ls/total': rec_loss.item(),
                    'Ls/reconstruction': rec_loss.item()
                }

                return rec_loss, status
            else:
                return rec_loss
        # wandb.log({"Per Sequence Reconstruction Loss": reconstruction_loss})
        # print(f"Reconstruction loss: {rec_loss}")
        loss = rec_loss * self.loss_weight['reconstruction'] + loss
        # bce_loss = self.objective['BCE'](out_seg,gt_flow)
        # try:
        #     IOU_loss = self.objective['mask_iou'](out_seg, gt_flow)
        # except:
        #     IOU_loss = torch.tensor(0.0).cuda()
        #
        # mse = self.objective['MSE'](out_seg,gt_flow)
        #
        # # Weighted sum
        # loss = self.loss_weight['BCE'] * bce_loss + self.loss_weight['mask_iou'] * IOU_loss + self.loss_weight['MSE'] * mse

        if return_status:
            status['Ls/total'] = loss.item()
            status['Ls/reconstruction'] = rec_loss.item()

            return loss, status
        else:
            return loss

    def smooth_contours(self, mask, sigmaX=4):
        # Assume mask is a 2D numpy array of type float32 with values in range [0, 1]
        # Convert the mask to uint8 format with values in range [0, 255]
        mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        # Find the contours of the mask
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoothed_mask = np.zeros_like(mask_np)
        cv2.drawContours(smoothed_mask, contours, -1, (255), thickness=cv2.FILLED)

        smoothed_mask = cv2.GaussianBlur(smoothed_mask, (0, 0), sigmaX)
        _, smoothed_mask = cv2.threshold(smoothed_mask, 127, 1, cv2.THRESH_BINARY)

        return smoothed_mask.astype(np.float32)

    def smooth_mask_batch(self, batch_tensor, sigmaX=4):
        #  batch_tensor has shape (B, 1, H, W)
        smoothed_batch = []

        for mask_tensor in batch_tensor:
            smoothed_mask_np = self.smooth_contours(mask_tensor, sigmaX)
            smoothed_mask_tensor = torch.from_numpy(smoothed_mask_np).unsqueeze(0)
            smoothed_batch.append(smoothed_mask_tensor)

        smoothed_batch_tensor = torch.stack(smoothed_batch)

        return smoothed_batch_tensor.to(self.device)

    def find_centroid(self, mask):
        """Grab a mask, and then generate the centroid of that mask, the mask will have a dimension of (8, 1, 320, 320)"""
        rows, cols = torch.where(mask > 0)
        num_pixels = len(rows)

        centroid_x = torch.sum(cols).float() / num_pixels
        centroid_y = torch.sum(rows).float() / num_pixels

        return centroid_x, centroid_y

    def convert_cxcywh_2_x1y1wh(self,bbox):

        cx = bbox[0]
        cy = bbox[1]
        w = bbox[2]
        h = bbox[3]

        x1 = torch.round(cx - 0.5*w).int()
        y1 = torch.round(cy - 0.5*h).int()

        return (x1,y1,w,h)

