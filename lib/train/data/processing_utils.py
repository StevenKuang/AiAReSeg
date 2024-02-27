import math
import random

import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from torchvision.ops import masks_to_boxes

from lib.train.data import flow_utils

# Modified from the original test implementation
# Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
# Add a variable called att_mask for computing attention and positional encoding later


def sample_target(im, target_bb, search_area_factor, output_sz=None):
    """
    Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area.

    Args:
        im: cv image.
        target_bb: Target box [x, y, w, h].
        search_area_factor: Ratio of crop size to target size.
        output_sz (float): Size to which the extracted crop is resized (always square). If None, no resizing is done.

    Returns:
        cv image: Extracted crop.
        float: The factor by which the crop has been resized to make the crop size equal output_size.
    """

    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('ERROR: too small bounding box')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)

    # Deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        return im_crop_padded, resize_factor, att_mask
    else:
        return im_crop_padded, att_mask.astype(np.bool_), 1.0

def sample_image_seg(im, seg_mask=None, bbox=None, search_area_factor=None, output_sz=None, data_invalid=False):
    # im: ndarray (H,W,3)(600,800,3) range 0-255
    # im_crop_padded: ndarray (320,320,3) range 0,1
    # try:
    #     bbox = generate_bboxes(seg_mask.unsqueeze(0))
    # except:
    #     data_invalid = True
    #     return None, None, None, None, None, data_invalid, None

    if bbox is None:
        try:

            bbox = generate_bboxes(seg_mask[0].unsqueeze(0))

        except:
            data_invalid = True
            return None, None, None, None, None, data_invalid, None

        x1,y1,x2,y2 = bbox.squeeze(0).tolist()
        w = x2-x1
        h = y2-y1
        bbox = [x1,y1,w,h]

    if not isinstance(bbox, list):
        x1, y1, w, h = bbox.tolist()
        # w = x2-x1
        # h = y2-y1
    else:
        x, y, w, h = bbox

    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    # crop_sz = math.ceil(math.sqrt(w * h) * self.params.search_factor * (1.0 + (enlarge_factor * iter)))
    if crop_sz < 1:
        # raise Exception('ERROR: too small bounding box')
        data_invalid = True
        return None, None, None, None, None, data_invalid, None

    x1 = round(x1 + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y1 + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # transform the bounding box to the format of x1y1wh


    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # print the cropped region in format of x1y1wh
    print("Cropped region: ", x1 + x1_pad, y1 + y1_pad, x2 - x2_pad - (x1 + x1_pad), y2 - y2_pad - (y1 + y1_pad))

    # # Lets plot both the image and the bounding box
    # fig, ax = plt.subplots(1,1, figsize=(20,20))
    # ax.imshow(im)
    # rect = patches.Rectangle((int(bbox[0]), int(bbox[1])), int(bbox[2]), int(bbox[3]), linewidth=2,  edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    # #  plot the cropping box on top of im before and after expanding
    # rect2 = patches.Rectangle((int(x1 + x1_pad), int(y1 + y1_pad)), x2 - x2_pad - (x1 + x1_pad), y2 - y2_pad - (y1 + y1_pad), linewidth=2, edgecolor='b', facecolor='none')
    # ax.add_patch(rect2)
    # plt.show()

    # Crop the target mask as well
    if isinstance(seg_mask,list) or isinstance(seg_mask,tuple):
        seg_mask = seg_mask[0]

    im_mask_crop = seg_mask[..., y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    if isinstance(im, np.ndarray):
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)

    elif isinstance(im, torch.Tensor):
        im_crop_padded = torch.nn.functional.pad(im_crop, (y1_pad, y2_pad, x1_pad, x2_pad), value=0)

    if isinstance(seg_mask, np.ndarray):
        mask_crop_padded = cv.copyMakeBorder(im_mask_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)

    elif isinstance(seg_mask, torch.Tensor):
        mask_crop_padded = torch.nn.functional.pad(im_mask_crop, (y1_pad, y2_pad, x1_pad, x2_pad), value=0)

    # Attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if output_sz is not None:
        resize_factor_H = output_sz/H
        resize_factor_W = output_sz/W

        if isinstance(im_crop_padded, np.ndarray):
            im_crop_padded = cv.resize(im_crop_padded, (output_sz,output_sz))
            att_mask = cv.resize(att_mask,(output_sz, output_sz)).astype(np.bool_)

        elif isinstance(im_crop_padded, torch.Tensor):
            im_crop_padded = torch.nn.functional.interpolate(im_crop_padded, size=(output_sz, output_sz), mode='bilinear', align_corners=False)
            att_mask = torch.nn.functional.interpolate(im_crop_padded, size=(output_sz,output_sz), mode='bilinear', align_corners=False)


        if isinstance(mask_crop_padded, np.ndarray):
            mask_crop_padded = cv.resize(mask_crop_padded, (output_sz, output_sz))
        elif isinstance(mask_crop_padded, torch.Tensor):
            mask_crop_padded = torch.nn.functional.interpolate(mask_crop_padded.unsqueeze(0).unsqueeze(0), size=(output_sz, output_sz),
                                                               mode='bilinear', align_corners=False)

        return im_crop_padded, resize_factor_W, resize_factor_H, att_mask, mask_crop_padded, data_invalid, bbox
def sample_image_unsup_seg(im, flow=None, bbox=None, search_area_factor=None, output_sz=None, data_invalid=False):

    # try:
    #     bbox = generate_bboxes(flow.unsqueeze(0))
    # except:
    #     data_invalid = True
    #     return None, None, None, None, None, data_invalid, None

    if bbox is None:
        try:

            bbox = generate_bboxes(flow[0].unsqueeze(0))

        except:
            data_invalid = True
            return None, None, None, None, None, data_invalid, None

        x1,y1,x2,y2 = bbox.squeeze(0).tolist()
        w = x2-x1
        h = y2-y1
        bbox = [x1,y1,w,h]

    if not isinstance(bbox, list):
        x1, y1, w, h = bbox.tolist()
        # w = x2-x1
        # h = y2-y1
    else:
        x, y, w, h = bbox

    # # Lets plot both the image and the bounding box
    # fig, ax = plt.subplots(1,2, figsize=(20,20))
    # ax[0].imshow(im)
    #
    # rect = patches.Rectangle((int(x1), int(y1)), int(w), int(h), linewidth=2,  edgecolor='r', facecolor='none')
    # ax[0].add_patch(rect)



    # Crop image
    crop_sz = 320
    # crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        # raise Exception('ERROR: too small bounding box')
        data_invalid = True
        return None, None, None, None, None, data_invalid, None

    x1 = round(x1 + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y1 + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)


    # Crop target
    # im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    im_crop = im
    # # plot the cropped image
    # plt.imshow(im_crop)
    # plt.show()
    #
    # # Also plot the segmentation mask
    # flow_plot = flow[0].detach().cpu().int().numpy()
    # mask = np.repeat(flow_plot[:,:,np.newaxis],3,axis=2)
    # mask = np.where(mask==1.,[1.,0.,0.],[0.,0.,0.])
    # rect = patches.Rectangle((int(x1), int(y1)), crop_sz, crop_sz, linewidth=2, edgecolor='b', facecolor='none')
    # ax[1].add_patch(rect)
    # # output = im.copy().astype(float)
    # # output = cv.addWeighted(output,1.0,mask,0.5, 0.0, dtype=cv.CV_8U)
    #
    # ax[1].imshow(mask)
    #
    # plt.show()

    # Crop the target flow as well
    if isinstance(flow,list) or isinstance(flow,tuple):
        flow = flow[0]

    # im_flow_crop = flow[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    im_flow_crop = flow
    # # plot the cropped flow
    # rgb_flow = torch.tensor(flow_utils.flow2img(im_flow_crop.cpu().numpy())).float() / 255.0
    # plt.imshow(rgb_flow)
    # plt.show()

    # Pad
    if isinstance(im, np.ndarray):
        # im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
        im_crop_padded = im_crop

    elif isinstance(im, torch.Tensor):
        # this implementation is wrong
        flow_crop_padded = torch.nn.functional.pad(im_crop, (y1_pad, y2_pad, x1_pad, x2_pad), value=0)

    flow_crop_padded = None
    # convert flow to np.ndarray
    im_flow_crop = im_flow_crop.cpu().numpy()
    if isinstance(im_flow_crop, np.ndarray):
        flow_crop_padded = im_flow_crop
        # flow_crop_padded = cv.copyMakeBorder(im_flow_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    elif isinstance(im_flow_crop, torch.Tensor):
        # this implementation is wrong
        flow_crop_padded = torch.nn.functional.pad(im_flow_crop, (y1_pad, y2_pad, x1_pad, x2_pad), value=0)

    # # plot the cropped flow
    # rgb_flow = torch.tensor(flow_utils.flow2img(flow_crop_padded)).float() / 255.0
    # plt.imshow(rgb_flow)
    # plt.show()

    # Attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    # att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if output_sz is not None:
        resize_factor_H = output_sz/H
        resize_factor_W = output_sz/W

        if isinstance(im_crop_padded, np.ndarray):
            im_crop_padded = cv.resize(im_crop_padded, (output_sz,output_sz))
            att_mask = cv.resize(att_mask,(output_sz, output_sz)).astype(np.bool_)

        elif isinstance(im_crop_padded, torch.Tensor):
            im_crop_padded = torch.nn.functional.interpolate(im_crop_padded, size=(output_sz, output_sz), mode='bilinear', align_corners=False)
            att_mask = torch.nn.functional.interpolate(im_crop_padded, size=(output_sz,output_sz), mode='bilinear', align_corners=False)


        if isinstance(flow_crop_padded, np.ndarray):
            flow_crop_padded = cv.resize(flow_crop_padded, (output_sz, output_sz))
        elif isinstance(flow_crop_padded, torch.Tensor):
            flow_crop_padded = torch.nn.functional.interpolate(flow_crop_padded.unsqueeze(0), size=(output_sz, output_sz),
                                                               mode='bilinear', align_corners=False)

        # # plot the cropped flow
        # rgb_flow = torch.tensor(flow_utils.flow2img(flow_crop_padded)).float() / 255.0
        # plt.imshow(rgb_flow)
        # plt.show()

        # convert back to tensor
        flow_crop_padded = torch.tensor(flow_crop_padded)
        # flip the u channel of the flow
        # flow_crop_padded[:,:,0] = -flow_crop_padded[:,:,0]

        # # plot the cropped image and flow
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(im_crop_padded)
        # # plot the cropped flow
        # rgb_flow = torch.tensor(flow_utils.flow2img(flow_crop_padded.cpu().numpy())).float() / 255.0
        # axs[1].imshow(rgb_flow)
        # plt.show()


        return im_crop_padded, resize_factor_W, resize_factor_H, att_mask, flow_crop_padded, data_invalid, bbox
def generate_bboxes(mask):
    # # Generating bounding boxes from the segmentation mask for cropping
    # non_zero_indices = torch.where(mask)
    #
    # min_row = torch.min(non_zero_indices[0])
    # min_col = torch.min(non_zero_indices[1])
    # max_row = torch.max(non_zero_indices[0])
    # max_col = torch.max(non_zero_indices[1])
    #
    # bounding_box = torch.tensor([min_row,min_col,max_row,max_col])
    #
    # # Convert the bounding box into the format of x1y1wh
    # w = (max_col-min_col)
    # h = (max_row-min_row)
    # bounding_box = torch.tensor([min_row-0.5*w, min_col - 0.5*h, w, h])

    boxes = masks_to_boxes(mask)

    return boxes

def image_proc_seg(frames, masks=None, jittered_boxes= None, search_area_factor=None, output_sz=None):
    if jittered_boxes is not None:
        crops_resize_factors = [sample_image_seg(f, seg_mask=masks, bbox=b, search_area_factor=search_area_factor, output_sz=output_sz) for f,b in zip(frames, jittered_boxes)]
    elif jittered_boxes is None:
        crops_resize_factors = [
            sample_image_seg(f, seg_mask=masks, search_area_factor=search_area_factor, output_sz=output_sz) for
            f in frames]

    frames_resized, resize_factor_W, resize_factor_H, att_mask, seg_mask, data_invalid, bbox = zip(*crops_resize_factors)

    return frames_resized, att_mask, seg_mask, data_invalid, resize_factor_W, resize_factor_H, bbox

def image_proc_unsup_seg(frames, flows=None, jittered_boxes= None, search_area_factor=None, output_sz=None):
    if jittered_boxes is not None:
        crops_resize_factors = [sample_image_unsup_seg(im=frame, flow=flow, bbox=b, search_area_factor=search_area_factor, output_sz=output_sz) for frame, flow, b in zip(frames, flows, jittered_boxes)]
    elif jittered_boxes is None:
        crops_resize_factors = [
            sample_image_unsup_seg(f, flow=flows, search_area_factor=search_area_factor, output_sz=output_sz) for
            f in frames]

    frames_resized, resize_factor_W, resize_factor_H, att_mask, seg_mask, data_invalid, bbox = zip(*crops_resize_factors)

    return frames_resized, att_mask, seg_mask, data_invalid, resize_factor_W, resize_factor_H, bbox

def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """
    Transform the box coordinates from the original image coordinates to the coordinates of the cropped image.

    Args:
        box_in: The box for which the coordinates are to be transformed.
        box_extract: The box about which the image crop has been extracted.
        resize_factor: The ratio between the original image scale and the scale of the image crop.
        crop_sz: Size of the cropped image.

    Returns:
        torch.Tensor: Transformed coordinates of box_in.
    """

    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz):
    """
    For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the coordinates of
    the box box_gt are transformed to the image crop coordinates.

    Args:
        frames: List of frames.
        box_extract: List of boxes of same length as frames. The crops are extracted using anno_extract.
        box_gt: List of boxes of same length as frames. The coordinates of these boxes are transformed from
                image coordinates to the crop coordinates.
        search_area_factor: The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz: The size to which the extracted crops are resized.

    Returns:
        list: List of image crops.
        list: box_gt location in the crop coordinates.
    """

    crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                            for f, a in zip(frames, box_extract)]
    frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)

    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # Find the bb location in the crop
    # Note that here we use normalized coord
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor,
                          normalize=False) -> torch.Tensor:
    """
    Transform the box coordinates from the original image coordinates to the coordinates of the cropped image.

    Args:
        box: The box for which the coordinates are to be transformed.
        crop_box: Bounding box defining the crop in the original image.
        crop_sz: Size of the cropped image.

    Returns:
        torch.Tensor: Transformed coordinates of box_in.
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def iou(reference, proposals):
    """
    Compute the IoU between a reference box with multiple proposal boxes.

    Args:
        reference: Tensor of shape (1, 4).
        proposals: Tensor of shape (num_proposals, 4).

    Returns:
        torch.Tensor: Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """

    # Intersection box
    tl = torch.max(reference[:, :2], proposals[:, :2])
    br = torch.min(reference[:, :2] + reference[:, 2:], proposals[:, :2] + proposals[:, 2:])
    sz = (br - tl).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = reference[:, 2:].prod(dim=1) + proposals[:, 2:].prod(dim=1) - intersection

    return intersection / union


def rand_uniform(a, b, shape=1):
    """
    Sample numbers uniformly between a and b.

    Args:
        a: Lower bound.
        b: Upper bound.
        shape: Shape of the output tensor.

    Returns:
        torch.Tensor: Tensor of shape=shape.
    """

    return (b - a) * torch.rand(shape) + a


def perturb_box(box, min_iou=0.5, sigma_factor=0.1):
    """
    Perturb the input box by adding gaussian noise to the co-ordinates.

    Args:
        box: Input box.
        min_iou: Minimum IoU overlap between input box and the perturbed box.
        sigma_factor: Amount of perturbation, relative to the box size. Can be either a single element,
                      or a list of sigma_factors, in which case one of them will be uniformly sampled.
                      Further, each of the sigma_factor element can be either a float, or a tensor of
                      shape (4,) specifying the sigma_factor per co-ordinate.

    Returns:
        torch.Tensor: The perturbed box.
    """

    if isinstance(sigma_factor, list):
        # If list, sample one sigma_factor as current sigma factor
        c_sigma_factor = random.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, torch.Tensor):
        c_sigma_factor = c_sigma_factor * torch.ones(4)

    perturb_factor = torch.sqrt(box[2] * box[3]) * c_sigma_factor

    # Multiple tries to ensure that the perturbed box has iou > min_iou with the input box
    for _ in range(10):
        c_x = box[0] + 0.5 * box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = random.gauss(c_x, perturb_factor[0])
        c_y_per = random.gauss(c_y, perturb_factor[1])

        w_per = random.gauss(box[2], perturb_factor[2])
        h_per = random.gauss(box[3], perturb_factor[3])

        box_per = torch.Tensor([c_x_per - 0.5 * w_per, c_y_per - 0.5 * h_per, w_per, h_per])

        box_iou = iou(box.view(1, 4), box_per.view(1, 4))

        # If there is sufficient overlap, return
        if box_iou > min_iou:
            return box_per, box_iou

        # Else, reduce the perturb factor
        perturb_factor *= 0.9

    return box_per, box_iou


def flows_to_boxes(flows: torch.Tensor) -> torch.Tensor:
    """
    Convert optical flow to bounding boxes.

    Args:
        flows: Optical flow tensor of shape (num_flows, H, W, 2).

    Returns:
        torch.Tensor: Bounding boxes of shape (num_flows, 4).
    """
    num_flows = flows.shape[0]
    frame_height, frame_width = flows[0].shape[:2]

    # we only consider flow larger than a threshold to be significant,
    # if a change in one direction is too small, we set it to 0
    # temp_flows = flows.clone()
    # threshold = 0.1
    # temp_flows[temp_flows.abs() < threshold] = 0

    # # debug
    # temp_flow_before = flow_utils.flow2img(flows[0].cpu().numpy())
    # before = torch.tensor(temp_flow_before).float() / 255.0
    #
    # temp_flows_after = flow_utils.flow2img(temp_flows[0].cpu().numpy())
    # after = torch.tensor(temp_flows_after).float() / 255.0
    #
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(before)
    # axs[0].set_title('Before')
    # axs[1].imshow(after)
    # axs[1].set_title('After')
    # plt.show()

    # Get the bounding box
    bounding_boxes = torch.zeros((num_flows, 4), device=flows.device, dtype=torch.float)
    threshold = 0.3
    for index, flow in enumerate(flows):
        v, u, _ = torch.where(torch.abs(flow) > threshold)
        # print('u:' + str(u.shape) + 'v:' + str(v.shape))
        if u.shape[0] == 0 or v.shape[0] == 0:
            return bounding_boxes
        bounding_boxes[index, 0] = torch.min(u)
        bounding_boxes[index, 1] = torch.min(v)
        bounding_boxes[index, 2] = torch.max(u)
        bounding_boxes[index, 3] = torch.max(v)

        bounding_boxes[index] = expand_box_edge(bounding_boxes[index], frame_height, frame_width, 0.2)

    # visualize the bounding box
    # visualize_flow_bbox(bounding_boxes, flows, 'xyxy')

    return bounding_boxes


def expand_box_edge(bounding_box, frame_height, frame_width, factor = 0.1):
    new_bounding_box = bounding_box.clone()
    width = bounding_box[2] - bounding_box[0]
    height = bounding_box[3] - bounding_box[1]
    min_width, min_height = int(factor * frame_width), int(factor * frame_height)
    if width >= min_width and height >= min_height:
        return bounding_box

    # expand the box
    if width < min_width:
        diff = min_width - width
        new_bounding_box[0] = max(0, new_bounding_box[0] - int(diff / 2))
        diff = diff - (bounding_box[0] - new_bounding_box[0])
        new_bounding_box[2] = min(frame_width, new_bounding_box[2] + int(diff))
    if height < min_height:
        diff = min_height - height
        new_bounding_box[1] = max(0, new_bounding_box[1] - int(diff / 2))
        diff = diff - (bounding_box[1] - new_bounding_box[1])
        new_bounding_box[3] = min(frame_height, new_bounding_box[3] + int(diff))
    return new_bounding_box


def visualize_flow_bbox(bounding_boxes, flows, mode='xywh'):
    for index, flow in enumerate(flows):
        temp_flow_before_raw = flow.cpu().numpy()
        temp_flow_before = flow_utils.flow2img(temp_flow_before_raw)
        before = torch.tensor(temp_flow_before).float() / 255.0
        # add the bounding box
        fig, ax = plt.subplots(1, 1)
        ax.imshow(before)
        if mode == 'xywh':
            rect = patches.Rectangle((bounding_boxes[index][0].item(), bounding_boxes[index][1].item()),
                                     bounding_boxes[index][2].item(), bounding_boxes[index][3].item(),
                                     linewidth=1, edgecolor='r', facecolor='none')
        elif mode == 'xyxy':
            rect = patches.Rectangle((bounding_boxes[index][0].item(), bounding_boxes[index][1].item()),
                                     bounding_boxes[index][2].item() - bounding_boxes[index][0].item(),
                                     bounding_boxes[index][3].item() - bounding_boxes[index][1].item(),
                                     linewidth=1, edgecolor='r', facecolor='none')
        else:
            raise ValueError('Invalid mode')
        ax.add_patch(rect)
        # print the bounding box in the image
        # ax.text(bounding_boxes[index][0].item(), bounding_boxes[index][1].item(), 'x1y1', color='red')
        plt.show()

