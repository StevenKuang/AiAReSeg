import os

import cv2
import numpy
import torch

from lib.models.aiareseg import build_aiareseg
from lib.test.tracker.utils import Preprocessor, Proprocessor_Seg
from lib.test.tracker.basetracker import BaseTracker
from lib.train.data.processing_utils import sample_target, transform_image_to_crop, image_proc_seg
from lib.utils.box_ops import clip_box
from lib.utils.merge import merge_feature_sequence
from torchvision.ops import masks_to_boxes
import math
from monai.losses import DiceLoss
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# For debugging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class AIARESEG(BaseTracker):
    def __init__(self, params, dataset_name):
        super(AIARESEG, self).__init__(params)
        network = build_aiareseg(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint)['net'], strict=True)
        self.cfg = params.cfg
        self.net = network.cuda()
        # # torch 2.0 feature
        # self.net = torch.compile(self.net)

        self.net.eval()
        self.preprocessor = Preprocessor()
        self.proprocessor_seg = Proprocessor_Seg()
        self.state = None
        # For debug
        self.debug = False
        self.frame_id = 0
        self.last_valid_frame = 0
        # Set the hyper-parameters
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.HYPER, DATASET_NAME):
            self.cache_siz = self.cfg.TEST.HYPER[DATASET_NAME][0]
            self.refer_cap = 1 + self.cfg.TEST.HYPER[DATASET_NAME][1]
            self.threshold = self.cfg.TEST.HYPER[DATASET_NAME][2]
        else:
            self.cache_siz = self.cfg.TEST.HYPER.DEFAULT[0]
            self.refer_cap = 1 + self.cfg.TEST.HYPER.DEFAULT[1]
            self.threshold = self.cfg.TEST.HYPER.DEFAULT[2]
        if self.debug:
            self.save_dir = 'debug'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # For save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.save_all_masks = params.save_all_masks

        self.dice = DiceLoss(reduction='none')
        self.distance_threshold = 10.0

    def initialize(self, image, info: dict, seq_name: str = None, segmentation: bool = False, unsupervised: bool = False):

        # First perform cropping and generate the masks
        if segmentation == True:
            if len(info) > 1:   # have init_bbox
                refer_crop, refer_att_mask, refer_seg_mask, data_invalid, resize_factor_W, resize_factor_H, bbox = image_proc_seg(
                    image,
                    masks=info['init_mask'],
                    jittered_boxes=[info['init_bbox']],
                    search_area_factor=self.params.search_factor,
                    output_sz=self.params.search_size)
                # refer_crop, refer_att_mask, refer_seg_mask, data_invalid, resize_factor_W, resize_factor_H, bbox = image_proc_unsup_seg(image,
                #                                                           masks=info['init_mask'],
                #                                                           search_area_factor=self.params.search_factor,
                #                                                           output_sz=self.params.search_size)
            else:
                # The bbox here is the uncropped initial bounding box
                refer_crop, refer_att_mask, refer_seg_mask, data_invalid, resize_factor_W, resize_factor_H, bbox = image_proc_seg(image,
                                                                          masks=info['init_mask'],
                                                                          search_area_factor=self.params.search_factor,
                                                                          output_sz=self.params.search_size)


            if data_invalid[0] == True:
                return True

            self.feat_size = self.params.search_size // 16
            refer_img = self.proprocessor_seg.process(refer_crop, refer_att_mask)

            print("sampling complete")

        else:
            # Forward the long-term reference once
            refer_crop, resize_factor, refer_att_mask = sample_target(image, info['init_bbox'], self.params.search_factor,
                                                                      output_sz=self.params.search_size)

            refer_box = transform_image_to_crop(torch.Tensor(info['init_bbox']), torch.Tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.search_size, self.params.search_size]),
                                                normalize=True)

            self.feat_size = self.params.search_size // 16
            refer_img = self.preprocessor.process(refer_crop, refer_att_mask)

            print("sampling complete")


        with torch.no_grad():
            # The reference dictionary contains info about Channel160, channel80, channel40, channel20
            refer_back = self.net.forward_backbone(refer_img)

            refer_dict_list = [refer_back]
            refer_dict = merge_feature_sequence(refer_dict_list)
            refer_mem = self.net.transformer.run_encoder(refer_dict['feat'], refer_dict['mask'], refer_dict['pos'],
                                                         refer_dict['inr'])

        if segmentation == True:

            target_region = torch.nn.functional.interpolate(refer_seg_mask[0], size=(self.feat_size,self.feat_size), mode='bilinear', align_corners=False)
            target_region = target_region.view(self.feat_size * self.feat_size, -1)
            background_region = 1 - target_region
            refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
            embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                                   dim=0).unsqueeze(0)

        else:
            target_region = torch.zeros((self.feat_size, self.feat_size))
            x, y, w, h = (refer_box * self.feat_size).round().int()
            target_region[max(y, 0):min(y + h, self.feat_size), max(x, 0):min(x + w, self.feat_size)] = 1
            target_region = target_region.view(self.feat_size * self.feat_size, -1)
            background_region = 1 - target_region
            refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
            embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                                   dim=0).unsqueeze(0)


        self.refer_mem_cache = [refer_mem]
        self.refer_emb_cache = [torch.bmm(refer_region, embed_bank).transpose(0, 1)]
        self.refer_pos_cache = [refer_dict['inr']]
        self.refer_msk_cache = [refer_dict['mask']]

        #NEW: Adding cache about each of the reference dictonary
        self.refer_temporal_cache = [refer_back]

        self.refer_mem_list = []
        for _ in range(self.refer_cap):
            self.refer_mem_list.append(self.refer_mem_cache[0])
        self.refer_emb_list = []
        for _ in range(self.refer_cap):
            self.refer_emb_list.append(self.refer_emb_cache[0])
        self.refer_pos_list = []
        for _ in range(self.refer_cap):
            self.refer_pos_list.append(self.refer_pos_cache[0])
        self.refer_msk_list = []
        for _ in range(self.refer_cap):
            self.refer_msk_list.append(self.refer_msk_cache[0])
        self.refer_temporal_dict = {}
        for _ in range(self.refer_cap):
            self.refer_temporal_dict[f"{_}"] = self.refer_temporal_cache[0]

        if segmentation == True:
            self.state = info['init_mask']

        else:
            # Save states
            self.state = info['init_bbox']
            if self.save_all_boxes:
                # Save all predicted boxes
                all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
                return {'all_boxes': all_boxes_save}

    def track(self, image, info: dict = None, seq_name: str = None, segmentation: bool = None):
        H, W, _ = image[0].shape
        self.frame_id += 5
        # debug
        print(f"Frame {self.frame_id}" + f"\tLast valid {self.last_valid_frame}")
        # Get the t-th search region
        if self.frame_id == 1:
            self.init_mask = self.state

        no_mask = True
        iter = 0
        valid_mask_count_req = 4 # minimum valid masks needed to present to form a merged mask # 2
        max_mask_search_iter = 10  # for the first hit  # 10 # 5 for real data
        addi_mask_search_overhead = 5   # chances for finding additional masks
        out_full_mask = None
        first_valid_iter = None
        enlarge_factor = 0.5
        self.params.search_factor = 1.5  # 1.5 as start may be too big
        while no_mask == True:
            if segmentation == True:
                if (0 <= iter < ((addi_mask_search_overhead + first_valid_iter) if first_valid_iter is not None else max_mask_search_iter)):
                    boxes = masks_to_boxes(self.state[0].unsqueeze(0)).squeeze(0).tolist()
                    boxes = self.x1y1x2y2_to_x1y1wh(boxes)
                    # print(f"Multishot, iter {iter}")
                    # print(boxes)
                    if boxes[2] < 50 and torch.sum(self.state[0]) < 2000:
                        diff = 50 - boxes[2]
                        boxes[0] -= diff // 2
                        boxes[2] = 50
                    if boxes[3] < 50 and torch.sum(self.state[0]) < 2000:
                        diff = 50 - boxes[3]
                        boxes[1] -= diff // 2
                        boxes[3] = 50
                    boxes = [torch.tensor(boxes)]
                    invalid_frame_tolerance = 3
                    frame_diff = 0
                    frame_diff_bloom = 0
                    while frame_diff + invalid_frame_tolerance < self.frame_id - self.last_valid_frame:
                        # boxes = self.move_bbox_towards_center(boxes, W, H, 5)
                        if self.frame_id - self.last_valid_frame > 5:
                            frame_diff_bloom += 0.2
                        frame_diff_bloom += 0.1
                        print("bloom search factor")
                        frame_diff += 1

                    # enlarged_search_area = self.params.search_factor * (1.0 + (0.01 * iter))
                    enlarged_search_area = (self.params.search_factor + frame_diff_bloom) * (1.0 + (enlarge_factor * iter))
                    # print(f"enlarged_search_area: {enlarged_search_area}")
                    print(f"iter {iter}/{((addi_mask_search_overhead + first_valid_iter) if first_valid_iter is not None else max_mask_search_iter)}")
                    search_crop, search_att_mask, search_seg_mask, data_invalid, resize_factor_W, resize_factor_H, bbox = image_proc_seg(image,
                                                                                  masks=self.state,
                                                                                  jittered_boxes=boxes,
                                                                                  search_area_factor=enlarged_search_area,
                                                                                  output_sz=self.params.search_size)
                    # print(f"Search area, iter {iter}")
                    # print(bbox)
                    if bbox != None:
                        bbox = (bbox[0].tolist(),)

                else:
                    no_mask = False
                    search_crop = (torch.zeros(size=(self.params.search_size,self.params.search_size,3)),)
                    search_att_mask = (torch.zeros(size=(self.params.search_size,self.params.search_size)),)
                    search_seg_mask = (torch.zeros(size=(1,1,self.params.search_size,self.params.search_size)),)
                    data_invalid = (True,)
                    resize_factor_W, resize_factor_H = W/self.params.search_size, H/self.params.search_size
                    bbox = [0,0,0,0]

                if data_invalid[0] == True and iter<=100:
                    iter += 1
                    data_invalid = (True,)
                    continue

                elif data_invalid[0] == True and iter>=100:
                    print("Problem")


                search_img = self.proprocessor_seg.process(search_crop, search_att_mask)

            if segmentation == False:

                search_crop, resize_factor, search_att_mask = sample_target(image, self.state, self.params.search_factor,
                                                                            output_sz=self.params.search_size)  # (x1, y1, w, h)
                search_img = self.preprocessor.process(search_crop, search_att_mask)


            search_dict_list = []
            with torch.no_grad():

                search_back = self.net.forward_backbone(search_img)

                search_back_short = {k: search_back[k] for i,k in enumerate(search_back) if i < 4}
                search_dict_list.append(search_back_short)
                search_dict = merge_feature_sequence(search_dict_list)

                # ########Plotting attention maps for debugging##########
                # # We plot the original, and then all of the attention maps in subsequent layers
                # search_img_viz = torch.tensor(search_crop[0]).unsqueeze(0).permute(0, 3, 1, 2)
                # # Denorm
                # search_img_copy = search_img_viz.permute(0, 2, 3, 1).detach().cpu()  # (b,320,320,3)
                # mean = torch.tensor([0.485, 0.465, 0.406])
                # std = torch.tensor([0.229, 0.224, 0.225])
                # search_img_denorm = search_img_copy
                # # search_img_denorm = (search_img_copy * std) + mean
                #
                # copy_src = search_dict['feat']
                # copy_src = copy_src.permute(1, 2, 0).view(-1, 256, 20, 20)
                # # gt_flow = data['search_flow'].squeeze(0).permute(0, 3, 1, 2)
                #
                # for i in range(copy_src.shape[0]):
                #     plot_img = copy_src[i, 0, ...].detach().cpu().numpy().astype(float)
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
                #
                # ########Plotting for debugging##########

                # Run the transformer
                out_embed, search_mem, pos_emb, key_mask = self.net.forward_transformer(search_dic=search_dict,
                                                                                        refer_mem_list=self.refer_mem_list,
                                                                                        refer_emb_list=self.refer_emb_list,
                                                                                        refer_pos_list=self.refer_pos_list,
                                                                                        refer_msk_list=self.refer_msk_list)

                if segmentation==True:
                    out_seg = self.net.forward_segmentation(out_embed, search_outputs=search_back, reference_outputs=self.refer_temporal_dict)

                else:
                    # Forward the corner head
                    out_dict, outputs_coord = self.net.forward_box_head(out_embed)
                    # out_dict: (B, N, C), outputs_coord: (1, B, N, C)

                    pred_iou = self.net.forward_iou_head(out_embed, outputs_coord.unsqueeze(0).unsqueeze(0))


            # Get the final result

            if segmentation == True:

                # Processing 1: Perform thresholding, any mask value < 0.5 is filtered out
                # you would only get her if iter is within the allowed range
                out_temp_mask = None
                if valid_mask_count_req > 0:   # we still need more mask
                    out_temp_mask = out_seg.squeeze(0).squeeze(0)
                    out_temp_mask = torch.where(out_temp_mask < 0.5, 0.0, 1.0)
                    check_sum = torch.sum(out_temp_mask)
                    if check_sum >= 10 * 20:    # mask big enough
                        if out_full_mask is None:   # if no mask yet
                            first_valid_iter = iter
                            resize_factors = [resize_factor_H, resize_factor_W]
                            # set initial mask
                            out_full_mask = self.map_mask_back(resize_factors=resize_factors, mask=out_temp_mask, bbox=bbox,
                                                               im_H=H, im_W=W, search_area_factor=enlarged_search_area)
                            valid_mask_count_req -= 1

                            # # plot the search_crop against the out_temp_mask
                            # rgb_mask_cropped = torch.tensor(out_temp_mask).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                            # rgb_mask_cropped = rgb_mask_cropped.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8) * 255
                            # overlay = cv2.addWeighted(search_crop[0], 1, rgb_mask_cropped[0], 0.5, 0)
                            # plt.figure(figsize=(10, 10))
                            # plt.title(f"Iter: {iter}, Search area factor: {enlarged_search_area}, bbox: {bbox}")
                            # plt.imshow(overlay)

                            # # plot the mask
                            # rgb_mask = torch.tensor(out_full_mask).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                            # rgb_mask = rgb_mask.permute(0, 2, 3, 1).detach().cpu().numpy()
                            # plt.imshow(rgb_mask[0])
                            # plt.title(f"First hit Iter {iter} of frame {self.frame_id}")
                            # plt.show()
                        else:   # we already have an initial mask and mask is big enough

                            # # plot the search_crop against the out_temp_mask
                            # rgb_mask_cropped = torch.tensor(out_temp_mask).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                            # rgb_mask_cropped = rgb_mask_cropped.permute(0, 2, 3, 1).detach().cpu().numpy().astype(
                            #     np.uint8) * 255
                            # overlay = cv2.addWeighted(search_crop[0], 1, rgb_mask_cropped[0], 0.5, 0)
                            # plt.figure(figsize=(10, 10))
                            # plt.title(f"Iter: {iter}, Search area factor: {enlarged_search_area}, bbox: {bbox}")
                            # plt.imshow(overlay)

                            resize_factors = [resize_factor_H, resize_factor_W]
                            before_merge_full_mask = self.map_mask_back(resize_factors=resize_factors,
                                                                        mask=out_temp_mask, bbox=bbox, im_H=H, im_W=W,
                                                                         search_area_factor=enlarged_search_area)

                            # # plot the mask
                            # rgb_mask = torch.tensor(out_full_mask).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                            # rgb_mask = rgb_mask.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8) * 255
                            # # change the color of rgb_mask to blue
                            # rgb_mask[0][:, :, 0] = 0
                            # rgb_mask[0][:, :, 1] = 0
                            # # overlay before_merge_full_mask as color
                            # rgb_before_merge_full_mask = torch.tensor(before_merge_full_mask).unsqueeze(0).unsqueeze(
                            #     0).repeat(1, 3, 1, 1).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8) * 255
                            # rgb_overlay = cv2.addWeighted(image[0], 1, rgb_before_merge_full_mask[0], 0.5, 0)
                            # rgb_overlay = cv2.addWeighted(rgb_overlay, 1, rgb_mask[0], 0.5, 0)
                            # plt.figure(figsize=(10, 10))
                            # plt.imshow(rgb_overlay)
                            # plt.title(f"Iter {iter} of frame {self.frame_id}")
                            # plt.show()

                            # merge
                            out_full_mask = torch.max(out_full_mask, before_merge_full_mask)
                            valid_mask_count_req -= 1

                        if valid_mask_count_req > 0:    # we still need more mask
                            iter += 1
                            continue
                        else:   # we have found enough masks
                            no_mask = False
                            self.last_valid_frame = self.frame_id
                    else:   # mask too small or no mask
                        # check if it's the last iteration
                        if iter == (((addi_mask_search_overhead + first_valid_iter) if first_valid_iter is not None else max_mask_search_iter) - 1) and out_full_mask is not None:
                            no_mask = False
                            self.last_valid_frame = self.frame_id
                        else:
                            iter += 1
                            continue
                else:   # we have found enough masks
                    no_mask = False
                    self.last_valid_frame = self.frame_id

                # Processing 2: Filter out smaller regions in the mask
                # # plot the final mask before extraction
                # check_sum = torch.sum(out_full_mask)
                # rgb_mask = torch.tensor(out_full_mask).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                # rgb_mask = rgb_mask.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8) * 255
                # rgb_overlay = cv2.addWeighted(image[0], 1, rgb_mask[0], 0.5, 0)
                # plt.figure(figsize=(10, 10))
                # plt.imshow(rgb_overlay)
                # plt.title(
                #     f"Final mask at Iter {iter} of frame {self.frame_id}, check_sum: {check_sum}, search area factor: {enlarged_search_area}\n Before Trim")
                # plt.show()

                out_full_mask = self.extract_largest_component(out_full_mask, prev_mask=self.state[0], bbox=boxes[0])
                # Processing 3: update the state
                new_state = [out_full_mask]

                # plot the final mask
                check_sum = torch.sum(new_state[0])
                rgb_mask = torch.tensor(new_state[0]).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                rgb_mask = rgb_mask.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8) * 255
                rgb_overlay = cv2.addWeighted(image[0], 1, rgb_mask[0], 0.5, 0)
                plt.figure(figsize=(10, 10))
                plt.imshow(rgb_overlay)
                plt.title(f"Final mask at Iter {iter} of frame {self.frame_id}, check_sum: {check_sum}, search area factor: {enlarged_search_area}")
                plt.show()

                # Check the intersection
                dice_loss = self.dice(new_state[0], self.state[0])
                IOU = self.dice_loss_to_iou(dice_loss)

                # Please check the bounding box outcome of the new state, if that new outcome generates an empty bounding box, then do not update the state!
                try:
                    state_bbox = masks_to_boxes(new_state[0].unsqueeze(0)).squeeze(0).tolist()
                    state_bbox = self.x1y1x2y2_to_x1y1wh(state_bbox)
                    if (state_bbox[2] != 0) or (state_bbox[3] != 0):
                        self.state = new_state
                        centroid_current = self.calculate_centroid(new_state[0])
                        centroid_old = self.calculate_centroid(self.state[0])
                        euclidean_distance = self.euclidean_distance(centroid_old, centroid_current)
                    else:
                        euclidean_distance = 0
                except:
                    self.state = self.state
                    euclidean_distance = 0


            else:
                pred_boxes = out_dict['pred_boxes'].view(-1, 4)
                # Baseline: Take the mean of all predicted boxes as the final result
                pred_box = (pred_boxes.mean(
                    dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                # Get the final box result
                self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

                predicted_iou = pred_iou['pred_iou'][0][0][0].item()

            # ax.add_patch(plt.Rectangle((self.state[0], self.state[1]), self.state[2], self.state[3], fill=False, color=[0.000, 0.447, 0.741], linewidth=3))
        if data_invalid[0] == True:
            return True

        if segmentation == True:

            if IOU > 0.7:
                # We do not wish to use the IOU as a method of keeping the references, instead,we will store every reference mask possible

                # Then we use all of the reference masks for the next prediction

                # This may be demanding on the computation power of the workstation, but we can reduce the buffer sizes if necessary
                self.feat_size = self.params.search_size // 16

                if len(self.refer_mem_cache) == self.cache_siz:
                    _ = self.refer_mem_cache.pop(1)
                    _ = self.refer_emb_cache.pop(1)
                    _ = self.refer_pos_cache.pop(1)
                    _ = self.refer_msk_cache.pop(1)
                    # New
                    _ = self.refer_temporal_cache.pop(1)

                # The target regions are updated based on previous segementation masks instead
                target_region = torch.nn.functional.interpolate(out_full_mask.unsqueeze(0).unsqueeze(0), size=(self.feat_size,self.feat_size), mode='bilinear', align_corners=False)
                target_region = target_region.view(self.feat_size * self.feat_size, -1)
                background_region = 1 - target_region
                refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
                embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight], dim=0).unsqueeze(0)
                new_emb = torch.bmm(refer_region, embed_bank).transpose(0, 1)

                self.refer_mem_cache.append(search_mem)
                self.refer_emb_cache.append(new_emb)
                self.refer_pos_cache.append(pos_emb)
                self.refer_msk_cache.append(key_mask)
                # new
                self.refer_temporal_cache.append(search_back)

                self.refer_mem_list = [self.refer_mem_cache[0]]
                self.refer_emb_list = [self.refer_emb_cache[0]]
                self.refer_pos_list = [self.refer_pos_cache[0]]
                self.refer_msk_list = [self.refer_msk_cache[0]]
                # new
                self.refer_temporal_dict = {}
                self.refer_temporal_dict['0'] = self.refer_temporal_cache[0]

                max_idx = len(self.refer_mem_cache) - 1
                ensemble = self.refer_cap - 1

                for part in range(ensemble):
                    temp = max_idx * (part + 1) // ensemble
                    dict_idx = part + 1
                    self.refer_mem_list.append(self.refer_mem_cache[max_idx * (part + 1) // ensemble])
                    self.refer_emb_list.append(self.refer_emb_cache[max_idx * (part + 1) // ensemble])
                    self.refer_pos_list.append(self.refer_pos_cache[max_idx * (part + 1) // ensemble])
                    self.refer_msk_list.append(self.refer_msk_cache[max_idx * (part + 1) // ensemble])
                    self.refer_temporal_dict[f'{dict_idx}'] = self.refer_temporal_cache[max_idx * (part + 1) // ensemble]

        else:
            # Update state
            if predicted_iou > self.threshold:
                if len(self.refer_mem_cache) == self.cache_siz:
                    _ = self.refer_mem_cache.pop(1)
                    _ = self.refer_emb_cache.pop(1)
                    _ = self.refer_pos_cache.pop(1)
                    _ = self.refer_msk_cache.pop(1)
                target_region = torch.zeros((self.feat_size, self.feat_size))
                x, y, w, h = (outputs_coord[0] * self.feat_size).round().int()
                target_region[max(y, 0):min(y + h, self.feat_size), max(x, 0):min(x + w, self.feat_size)] = 1
                target_region = target_region.view(self.feat_size * self.feat_size, -1)
                background_region = 1 - target_region
                refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
                embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                                       dim=0).unsqueeze(0)
                new_emb = torch.bmm(refer_region, embed_bank).transpose(0, 1)
                self.refer_mem_cache.append(search_mem)
                self.refer_emb_cache.append(new_emb)
                self.refer_pos_cache.append(pos_emb)
                self.refer_msk_cache.append(key_mask)

                self.refer_mem_list = [self.refer_mem_cache[0]]
                self.refer_emb_list = [self.refer_emb_cache[0]]
                self.refer_pos_list = [self.refer_pos_cache[0]]
                self.refer_msk_list = [self.refer_msk_cache[0]]
                max_idx = len(self.refer_mem_cache) - 1
                ensemble = self.refer_cap - 1
                for part in range(ensemble):
                    temp = max_idx * (part + 1) // ensemble
                    self.refer_mem_list.append(self.refer_mem_cache[max_idx * (part + 1) // ensemble])
                    self.refer_emb_list.append(self.refer_emb_cache[max_idx * (part + 1) // ensemble])
                    self.refer_pos_list.append(self.refer_pos_cache[max_idx * (part + 1) // ensemble])
                    self.refer_msk_list.append(self.refer_msk_cache[max_idx * (part + 1) // ensemble])

        # For debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 255, 0), thickness=3)
            save_seq_dir = os.path.join(self.save_dir, seq_name)
            if not os.path.exists(save_seq_dir):
                os.makedirs(save_seq_dir)
            save_path = os.path.join(save_seq_dir, '%04d.jpg' % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            # Save all predictions
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N,)
            return {'target_bbox': self.state,
                    'all_boxes': all_boxes_save}

        if self.save_all_masks:
            return {'target_mask': self.state}

        else:
            return {'target_bbox': self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_mask_back(self, mask, resize_factors, bbox, im_H, im_W, search_area_factor):

        # This is the bounding box of the previous frame's segmentation mask, not enlarged
        if isinstance(bbox, list) or isinstance(bbox, tuple):
            x1,y1,w,h = bbox[0]

        else:
            x1, y1, x2, y2 = bbox[0][0, ...].tolist()
            w = x2 - x1
            h = y2 - y1
        # if iter == 0:
        #     # You may want to enlarge it in order to make the sizes match
        #     crop_sz = math.ceil(math.sqrt(w * h) * self.params.search_factor)
        # else:
        # search_area_factor = self.params.search_factor * (1.0 + (enlarge_factor*iter))
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)


        x1_new = round(x1 + 0.5 * w - crop_sz * 0.5)
        x2_new = x1_new + crop_sz
        y1_new = round(y1 + 0.5 * h - crop_sz * 0.5)
        y2_new = y1_new + crop_sz

        x1_new_pad = max(0, -x1_new)
        x2_new_pad = max(x2_new - im_W + 1, 0)

        y1_new_pad = max(0, -y1_new)
        y2_new_pad = max(y2_new - im_H + 1, 0)

        padded_W = (x2_new - x2_new_pad) - (x1_new + x1_new_pad)
        padded_H = (y2_new - y2_new_pad) - (y1_new + y1_new_pad)

        # print("Cropped region: ", x1 + x1_pad, y1 + y1_pad, x2 - x2_pad - (x1 + x1_pad), y2 - y2_pad - (y1 + y1_pad))
        print("Scale up region: ", x1_new + x1_new_pad, y1_new + y1_new_pad, padded_W, padded_H, "\tsearch area factor: ", search_area_factor, "\tcrop_sz: ", crop_sz)
        # After enlarged, take the mask, shrink it to the size you desire

        mask = mask.squeeze(0).squeeze(0).detach().cpu().numpy()
        # first resize to crop_sz x crop_sz
        mask_orig_cropped = cv2.resize(mask, (crop_sz, crop_sz), interpolation=cv2.INTER_LINEAR)

        # then remove the padding
        mask_orig_cropped = mask_orig_cropped[y1_new_pad:y1_new_pad + padded_H, x1_new_pad:x1_new_pad + padded_W]
        mask_orig_cropped = torch.tensor(mask_orig_cropped)

        # mask_orig_cropped = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(padded_H, padded_W),
        #                                                     mode='bicubic', align_corners=True).squeeze(0).squeeze(0)

        # Once shrunk, create a tensor of zeros
        mask_orig = torch.zeros(size=(im_H, im_W))

        # Replace the patch with the mask
        mask_orig[(y1_new+y1_new_pad):(y2_new-y2_new_pad), (x1_new+x1_new_pad):(x2_new-x2_new_pad)] = mask_orig_cropped

        # check_sum = torch.sum(mask_orig)

        #ask_resized = torch.nn.interpolate(mask, size=(w,h), mode='bilinear', align_corners=False)

        #mask_original = torch.zeros()

        return mask_orig

    def map_mask_back_new(self, mask, previous_box, resize_factor, iteration, enlarge_factor):

        # Inputs:
        # Mask: The generated mask output
        # Previous box: The bounding box from the last output
        # Resize factor: This tells you what is the original size of the bounding boxes

        H, W = self.state[0].shape
        output_mask = torch.zeros(size=(H, W))

        previous_box = self.x1y1x2y2_to_x1y1wh(previous_box[0].squeeze(0).tolist())

        bbox = masks_to_boxes(mask.unsqueeze(0)) # Remember, this is in the cooridnates of the cropped frame

        x1,y1,x2,y2 = bbox.squeeze(0).tolist()
        w = x2-x1
        h = y2-y1
        cx = x1 + 0.5*w
        cy = y1 + 0.5*h
        bbox = [cx, cy, w, h]
        bbox = [elem/resize_factor[0] for elem in bbox]

        mapped_box = self.map_box_back_new(bbox, resize_factor[0], previous_box)

        # Find the center point
        cx_true = mapped_box[0] + 0.5*mapped_box[2]
        cy_true = mapped_box[1] + 0.5*mapped_box[3]

        x1_true = round(cx_true - 0.5*self.params.search_size/resize_factor[0])
        x2_true = round(x1_true + self.params.search_size/resize_factor[0])

        y1_true = round(cy_true - 0.5*self.params.search_size/resize_factor[0])
        y2_true = round(y1_true + self.params.search_size/resize_factor[0])


        # Find the actual size

        # We will enlarge the mapped box by the w*h*search_factor
        search_area_factor = self.params.search_factor * (1.0 + (enlarge_factor * iteration))
        # crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
        crop_size = math.ceil(mapped_box[2] * mapped_box[3] * search_area_factor)
        x1_l = round(mapped_box[0] + 0.5 * mapped_box[2] - crop_size * 0.5)
        x2_l = x1_l + crop_size

        y1_l = round(mapped_box[1] + 0.5 * mapped_box[3] - crop_size * 0.5)
        y2_l = y1_l + crop_size

        resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(round(self.params.search_size/resize_factor[0]), round(self.params.search_size/resize_factor[0])), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Replace the image with the actual crop

        try:
            output_mask[y1_true:y2_true, x1_true:x2_true] = resized_mask
        except:
            print(f"Size mismatch error, the mask size is {y2_true-y1_true},{x2_true-x1_true}, and the resized_mask is {round(self.params.search_size/resize_factor[0])}")




        return output_mask

    def map_box_back_new(self, pred_box: list, resize_factor: float, previous_box: list):
        cx_prev, cy_prev = previous_box[0] + 0.5 * previous_box[2], previous_box[1] + 0.5 * previous_box[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor # Half the actual size of the bounding box
        cx_real = cx + (cx_prev - half_side) # The origin of the previous frame, and then add the coordinates of the new frame
        cy_real = cy + (cy_prev - half_side) # So now we are in the coordinate system of the
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def dice_loss_to_iou(self, dice_loss):
        return 1 - dice_loss / (2 - dice_loss)

    def x1y1x2y2_to_x1y1wh(self, bbox):

        x1,y1,x2,y2 = bbox

        w = x2-x1
        h = y2-y1

        return [x1,y1,w,h]

    def _get_jittered_box(self,box,factor_scale, factor_center):
        """
        Jitter the input box.

        Args:
            box: Input bounding box.
            mode: String 'reference' or 'search' indicating reference or search data.

        Returns:
            torch.Tensor: Jittered box.
        """

        noise = torch.exp(torch.randn(2) * factor_scale)
        jittered_size = box[2:4] * noise
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(factor_center).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def calculate_centroid(self, mask):
        if isinstance(mask, numpy.ndarray):
            mask = torch.tensor(mask)
        total_ones = torch.sum(mask)
        y_moment, x_moment = self.calculate_moments(mask)
        centroid_y = y_moment / total_ones
        centroid_x = x_moment / total_ones
        return centroid_y.item(), centroid_x.item()

    def calculate_moments(self,mask):
        y_moment = torch.sum(torch.arange(mask.shape[0]).float().unsqueeze(1) * mask)
        x_moment = torch.sum(torch.arange(mask.shape[1]).float().unsqueeze(0) * mask)
        return y_moment, x_moment

    def euclidean_distance(self, elem1, elem2):
        output = math.sqrt((elem1[0]-elem2[0])**2 + (elem1[1]-elem2[1])**2)
        return output
    def move_bbox_towards_center(self, bbox, W, H, magnitude):
        # bbox is x1y1wh left upper corner, W,H are image dimension
        x1,y1,w,h = bbox[0]
        image_center_x, image_center_y = W / 2, H / 2

        bbox_center_x = x1 + w / 2
        bbox_center_y = y1 + h / 2

        move_x = magnitude if bbox_center_x < image_center_x else -magnitude
        move_y = magnitude if bbox_center_y < image_center_y else -magnitude

        # Move x1, y1 towards the image center , ensuring it doesn't go outside the image bounds
        new_x1 = min(max(x1 + move_x, 0), W - w)
        new_y1 = min(max(y1 + move_y, h), H - h)

        return [torch.tensor([new_x1, new_y1, w, h])]

    def extract_largest_component(self, mask, prev_mask=None, bbox=None):
        """
        Extract the largest connected component from a binary segmentation mask.

        :param mask: Binary mask as a numpy ndarray of shape (H, W).
        :return: Mask with only the largest connected component.
        """
        # Ensure mask is a binary image of type uint8
        mask = np.uint8(mask)
        # find the largest connected component on the previous mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(prev_mask.cpu().numpy().astype(np.uint8), 4, cv2.CV_32S)
        largest_label = np.argmax(stats[1:, -1]) + 1
        prev_mask_largest_region = torch.tensor(labels == largest_label).float()

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        valid_labels = [False] * num_labels
        need_pca = False
        keep_prev = False
        keep_prev_largest = False
        possible_labels = [i for i in range(1, num_labels)]
        # discard regions that are too small and does not overlap with previous mask
        for i in range(1, num_labels):
            if torch.sum(torch.tensor(labels == i).float()) < 50 and not torch.sum(prev_mask * torch.tensor(labels == i).float()) > 0:
                possible_labels.remove(i)

        if len(possible_labels) > 1:
            # Skip the background label at index 0
            valid_labels[0] = False
            sizes = stats[1:, -1]

            # Step 1: max overlap with previous mask, keep all components that has overlap with previous mask
            overlap_threshold = 0.4
            for i in possible_labels:
                # if torch.sum(prev_mask_largest_region * torch.tensor(labels == i).float()) > torch.sum(torch.tensor(labels == i).float()) * overlap_threshold:
                if torch.sum(prev_mask_largest_region * torch.tensor(labels == i).float()) > torch.sum(prev_mask_largest_region) * 0.5:
                    valid_labels[i] = True
                    # keep_prev = True
                    # keep_prev_largest = True

            if not any(valid_labels):
                need_pca = True

            prev_center = self.calculate_centroid(prev_mask)  # returns (y, x)
            # if torch.sum(prev_mask) <= 150:
            #     need_pca = False        # PCA unreliable on small blob
            #     centroids = np.zeros((num_labels, 2))
            #     dists = []
            #     for i in range(1, num_labels):
            #         if sizes[i - 1] < 50:
            #             continue
            #         centroids[i] = self.calculate_centroid(labels == i)
            #         dists.append(np.linalg.norm(centroids[i] - prev_center))
            #     # pick the closest one
            #     valid_labels[np.argmin(dists) + 1] = True
            #     keep_prev = True



            # Step 2: If there's no overlapping, pca on previous mask
            if need_pca:
                y, x = np.where(prev_mask == 1)
                coordinates = np.column_stack((x, y))
                pca = PCA(n_components=2)
                pca.fit(coordinates)
                components = pca.components_
                mean = pca.mean_
                # major axis
                major_axis = components[0]
                minor_axis = components[1]
                # veritcal mirror the major axis
                major_axis[1] = -major_axis[1]
                minor_axis[1] = -minor_axis[1]

                # find centeroids of connected components
                projected_len = []
                centroids = np.zeros((num_labels, 2))
                for i in possible_labels:
                    if sizes[i - 1] < 50:
                        continue
                    centroids[i] = self.calculate_centroid(labels == i)
                    # dir_vec = normalize([centroids[i] - prev_center]).ravel()
                    dir_vec = centroids[i] - prev_center
                    # project to major axis
                    # projected_len.append(np.abs(np.dot(dir_vec, major_axis))/np.linalg.norm(major_axis))
                    # project to minor axis
                    projected_len.append(np.abs(np.dot(dir_vec, minor_axis)))

                if not any(projected_len):
                    need_pca = False
                    keep_prev = True
                else:
                    valid_labels[np.argmax(projected_len) + 1] = True
                    # valid_labels[np.argmin(projected_len) + 1] = True
                    keep_prev_largest = True
                    keep_prev = True

            # Step 3: if there's more than 1 valid label, we remove all labels that have no overlap with previous bbox
            if sum(valid_labels) > 1:
                for i in possible_labels:
                    if valid_labels[i]:
                        if not torch.sum(torch.tensor(labels == i)[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]) > 0:
                            valid_labels[i] = False

        else:
            valid_labels[1] = True


        # Lets plot both the image and the bounding box
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        rgb_mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        rgb_mask = rgb_mask.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8) * 255
        # for i in range(1, num_labels):
        #     rgb_mask[0][labels == i] = [0, 0, 255]
        # plot previous mask
        rgb_mask[0][prev_mask == 1] = [0, 255, 0]   # green
        for i in possible_labels:
            if valid_labels[i]:
                rgb_mask[0][labels == i] = [0, 255, 255]
        # rgb_mask[0][labels == largest_label] = [0, 255, 255]    # teal
        ax.imshow(rgb_mask[0])
        rect = patches.Rectangle((int(bbox[0]), int(bbox[1])), int(bbox[2]), int(bbox[3]), linewidth=2, edgecolor='r',
                                 facecolor='none')
        if need_pca:
            plt.scatter(centroids[1:, 1], centroids[1:, 0], c='r', s=40)
            plt.scatter(prev_center[1], prev_center[0], c='b', s=40)
            plt.quiver(mean[0], mean[1], major_axis[0], major_axis[1], color='r', scale=5)  # red as major
            plt.quiver(mean[0], mean[1], minor_axis[0], minor_axis[1], color='g', scale=5)

        ax.add_patch(rect)
        plt.show()

        # Create a new mask for the largest connected component
        largest_component = np.zeros_like(mask)
        for i in possible_labels:
            if valid_labels[i]:
                largest_component[labels == i] = 1.0
        # largest_component[labels == largest_label] = 1.0
        largest_component = torch.tensor(largest_component).float()

        # If the largest component is too small, keep the previous mask
        if torch.sum(largest_component) < 200:
            keep_prev = True
        if keep_prev:
            if keep_prev_largest:
                largest_component = torch.max(largest_component, prev_mask_largest_region)
            else:
                largest_component = torch.max(largest_component, prev_mask)
        return largest_component
def get_tracker_class():
    return AIARESEG
