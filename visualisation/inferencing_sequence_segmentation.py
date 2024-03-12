import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import monai
import cv2 as cv

# In this script we will visualise the results after we pass the sequences through the AiA track framework
# The inferencing from the model will not be done in real time, we will implement that in another script instead
# In this script we will simply specify a sequence, and the visualise the bounding boxes one by one

def get_names(root_path, seq_no='51'):

    """
    Grabs the names of of all of the files under a root path
    """

    #root_path = _join_paths(root_path,seq_no)

    # dir_list = os.listdir(root_path)
    names = []
    paths = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            # print(os.path.join(path, name))
            # print(name)
            file_dir = os.path.join(os.path.basename(path),name)
            names.append(file_dir)
            paths.append(os.path.join(path, name))

    return names, paths

def _join_paths(root_path, seq_no='51'):

    """
    Joins the paths of the root with the specified sequence
    """

    return os.path.join(root_path,'Catheter',f'Catheter-{seq_no}','img')

def sort_seq_names(name):
    parts = name.split("/")
    return int(parts[-1][:-4])

def sort_numeric_part(name):
    # Split the filename into non-digit and digit parts
    name = name.split("/")[-1]

    # Extract the numeric part as an integer
    return int(name[:-4])

def open_mask(mask_path, seq_no='51'):

    """
    Outputs a list full of the ground truth bounding boxes for each of the images of the sequence
    """

    mask_folder_path = os.path.join(mask_path,f"label0031_%02d"%(int(seq_no)))

    names, files = get_names(mask_folder_path)
    mask_list = []
    final_elem = files[0].split('/')[-1]
    if len(final_elem)>5:
        files = sorted(files)
    else:
        files = sorted(files, key=sort_seq_names)


    for file in files:
        mask = torch.tensor(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        mask_list.append(mask)

    return torch.stack([mask for mask in mask_list], dim=0)

def open_model_mask(mask_path, seq_no='51'):
    mask_folder_path = os.path.join(mask_path,f"label0031_%02d"%(int(seq_no)))

    names, files = get_names(mask_folder_path)
    mask_list = []
    final_elem = files[0].split('/')[-1]

    files = sorted(files, key=sort_numeric_part)



    for file in files:
        mask = torch.tensor(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        mask_list.append(mask)

    return torch.stack([mask for mask in mask_list], dim=0)

def open_images(image_path, seq_no='51'):

    #image_folder_path = os.path.join(image_path, 'Catheter', f'Catheter-{seq_no}', 'img')
    image_folder_path = os.path.join(image_path, f"label0031_%02d"%(int(seq_no)))
    names, files = get_names(image_folder_path)
    image_list = []
    files = sorted(files)
    for file in files:
        image = torch.tensor(cv2.imread(file, cv2.IMREAD_COLOR))
        image_list.append(image)

    return torch.stack([image for image in image_list], dim=0)


def loop_image(image_tensor, gt_mask_tensor, model_mask_tensor, id):

    # Having three seperate tensors allows you to plot everything on top of one another by simply doing one loop

    B,H,W,C = image_tensor.shape

    for i in range(B):
        image = image_tensor[i,...].float().numpy().astype(float)/255.0 * 0.8
        gt_mask = gt_mask_tensor[i,...].float()
        model_mask = model_mask_tensor[i,...].float()


        # Next, we need to plot the image and the masks

        # fig, axs = plt.subplots(1,3,figsize=(20,20))
        #
        # # Plot the original image
        # axs[0].imshow(image)
        # # Plot the original image with the gt mask
        # gt_mask = gt_mask.permute(1,2,0).repeat(1,1,3).numpy()
        # gt_mask = np.where(gt_mask == 1, [0,255,0],[0,0,0]).astype(float) /255.0
        # output1 = image
        # output1 = cv2.addWeighted(output1, 1, gt_mask, 0.5, 0)
        # axs[1].imshow(output1)
        # # Plot the original image with the output mask
        # model_mask = model_mask.permute(1,2,0).repeat(1,1,3).numpy()
        # model_mask = np.where(model_mask == 1, [255,0,0],[0,0,0]).astype(float)/255.0
        # output2 = image
        # output2 = cv2.addWeighted(output2, 1, model_mask, 0.5, 0)
        # axs[2].imshow(output2)

        model_mask = model_mask.permute(1, 2, 0).repeat(1, 1, 3).numpy()
        model_mask = np.where(model_mask == 1, [255, 0, 0], [0, 0, 0]).astype(float) / 255.0
        gt_mask = gt_mask.permute(1, 2, 0).repeat(1, 1, 3).numpy()
        gt_mask = np.where(gt_mask == 1, [0, 255, 0], [0, 0, 0]).astype(float) / 255.0
        output2 = image
        output2 = cv2.addWeighted(output2, 1, model_mask, 0.5, 0)
        output2 = cv2.addWeighted(output2, 1, gt_mask, 0.5, 0)
        plt.figure(figsize=(15,15))
        ax = plt.gca()
        plt.imshow(output2)
        ax.set_axis_off()
        plt.show()


        # save_dir = f'/media/liming/Data/IDP/dataset/results/aligned_jit_step_560/{id}'
        # #
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # plt.savefig(os.path.join(save_dir,f'{i:06d}.png'))
        plt.close()


if __name__ == '__main__':

    gt_mask_path = '/media/liming/Data/IDP/dataset/us_simulation3_cactuss/rotations_transverse_31/filtered_masks'
    image_path = '/media/liming/Data/IDP/dataset/us_simulation3_cactuss/rotations_transverse_31/filtered'

    # model_mask_path = '/media/liming/Data/IDP/AiAProj/AiAReSeg/test/tracking_results/aiareseg/AiASeg_unsupervised/rotations_transverse_31/aligned_step_584'
    # model_mask_path = '/media/liming/Data/IDP/AiAProj/AiAReSeg/test/tracking_results/aiareseg/AiASeg_unsupervised/rotations_transverse_31/misaligned_0585'
    # model_mask_path = '/media/liming/Data/IDP/AiAProj/AiAReSeg/test/tracking_results/aiareseg/AiASeg_unsupervised/rotations_transverse_31/misaligned_0454'
    model_mask_path = '/media/liming/Data/IDP/AiAProj/AiAReSeg/test/tracking_results/aiareseg/AiASeg_unsupervised/rotations_transverse_31/aligned_jit_step_560'
    # model_mask_path = '/media/liming/Data/IDP/AiAProj/AiAReSeg/test/tracking_results/aiareseg/AiASeg_unsupervised/rotations_transverse_31/filtered'

    # phantom_list = ["02", "04", "05", "06", "07", "08"]
    # phantom_frame_trim = [0, 0, 0, 0, 0, 0]
    # chosen = 0
    # model_pred_folder_name = 'filtered'
    #
    # gt_mask_path = '/media/liming/Data/IDP/dataset/us_phantom/phantom_transverse_' + phantom_list[chosen] + '/filtered_masks'
    # image_path = '/media/liming/Data/IDP/dataset/us_phantom/phantom_transverse_' + phantom_list[chosen] + '/filtered'
    # model_mask_path = '/media/liming/Data/IDP/AiAProj/AiAReSeg/test/tracking_results/aiareseg/AiASeg_unsupervised/phantom_transverse_' + phantom_list[chosen] + '/' + model_pred_folder_name


    dice = monai.losses.DiceLoss(jaccard=False)
    mae = monai.metrics.MAEMetric()

    dice_list = []
    mae_list = []
    # count subfolders in model_mask_path
    folder_count = sum(os.path.isdir(os.path.join(model_mask_path, entry)) for entry in os.listdir(model_mask_path))
    for i in range(0, folder_count):
        seq_no = str(i)
        gt_mask_tensor = open_mask(gt_mask_path, seq_no=seq_no).unsqueeze(1)
        model_mask_tensor = open_model_mask(model_mask_path, seq_no=seq_no).unsqueeze(1)
        # model_mask_tensor = torch.roll(model_mask_tensor, shifts=10, dims=2)
        # model_mask_tensor = torch.roll(model_mask_tensor, shifts=10, dims=3)
        if gt_mask_tensor.max() == 2:
            gt_mask_np = gt_mask_tensor.numpy()
            # gt_mask_tensor of shape (B, C, H, W)
            # do some dialation on the catheter
            kernel = 2 * np.ones((10, 10), np.uint8)
            gt_mask_tensor = (gt_mask_np == 2).astype(np.uint8)  # Isolate the catheter
            # dilate the gt_mask_tensor using the kernel
            for j in range(gt_mask_tensor.shape[0]):
                gt_mask_tensor[j, 0, :, :] = cv2.dilate(gt_mask_tensor[j, 0, :, :], kernel, iterations=2)
            gt_mask_tensor = torch.tensor(gt_mask_tensor)
            # vertical flip gt_mask_tensor of shape (B, C, H, W)
            gt_mask_tensor = torch.flip(gt_mask_tensor, [2])
            # move the catheter up by 5 pixels
            gt_mask_tensor = torch.roll(gt_mask_tensor, shifts=-10, dims=2)

        # gt_mask_tensor = torch.where(gt_mask_tensor==2,1,0)
        image_tensor = open_images(image_path, seq_no=seq_no)
        if gt_mask_tensor.shape[0] > model_mask_tensor.shape[0]:
            frame_diff = gt_mask_tensor.shape[0] - model_mask_tensor.shape[0]
            # Pad the model mask tensor at the start
            model_mask_tensor = torch.cat([torch.zeros(frame_diff, *model_mask_tensor.shape[1:]), model_mask_tensor], dim=0)


        dice_list.append(dice(gt_mask_tensor, model_mask_tensor).item())
        mae_list.append(torch.mean(mae(gt_mask_tensor, model_mask_tensor)).item())

        loop_image(image_tensor, gt_mask_tensor, model_mask_tensor, id=i)
    dice_list = np.array(dice_list)
    average_dice = np.mean(dice_list)
    sd_dice = np.std(dice_list)

    mae_list = np.array(mae_list)
    average_mae = np.mean(mae_list)
    sd_mae = np.std(mae_list)

    print("Done")
    print(f"Dice_loss:{average_dice}")
    print(f"SD Dice_loss:{sd_dice}")
    print(f"MAE: {average_mae}")
    print(f"SD MAE: {sd_mae}")
    # Performance 0.0808, dice:0.9192 , average mae:0.002134
    # Catheter performance, dice: 0.831, average mae: 0.000138

    # TODO: Loop through each image, and then overlay the gt and the model outputs on the same image
    # misaligned 0585 -- 1 iter -- after shifting:
    # Dice_loss:0.44543371440433877
    # MAE: 0.003878319211807826
    # misaligned 0585 -- 2 iter -- after shifting:
    # Dice_loss: 0.37824380216075154
    # MAE: 0.00394323936269779

    # misaligned 0454 -- 1 iter -- after shifting:
    # Dice_loss: 0.4512748546534922
    # MAE: 0.0041396369657687926
    # misaligned 0454 -- 2 iter -- after shifting:
    # Dice_loss: 0.3769470280626925
    # MAE: 0.004078182061431751

    # aligned_step_0584 -- 1 iter:
    # Dice_loss: 0.6590016681428362
    # MAE: 0.003393401719998709

    # aligned_cosine_630 -- 1 iter:
    # Dice_loss: 0.7015915868076973
    # MAE: 0.0034247522353991105

    # aligned_jit_step_560 -- 1 iter:
    # Dice_loss:0.45517472800139974
    # MAE: 0.0035046001304158934

    # aligned_jit_step_560 -- 2 iter:
    # Dice_loss: 0.4211802001951671
    # MAE: 0.003954319072673991