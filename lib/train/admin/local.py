
# PATH = "/home/liming/Documents/AiAProj"
# PATH = "/mnt/c/Users/steve/OneDrive/Documents/12SemesterTUM/IDP/repos/AiAProj/"
PATH = "/media/liming/Data/IDP/AiAProj"
DATASET_ROOT = "/media/liming/Data/IDP/dataset"
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = PATH + '/AiAReSeg'  # Base directory for saving network checkpoints
        self.tensorboard_dir = self.workspace_dir  # Directory for tensorboard files
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'
        # self.lasot_dir = PATH + '/LaSOT'
        # self.got10k_dir = PATH + '/GOT10k/train'
        # self.trackingnet_dir = PATH + '/TrackingNet'
        # self.coco_dir = PATH + '/COCO'

        # self.lasot_dir = '/media/atr17/HDD Storage/Datasets_Download/LaSOT/LaSOT'
        # self.got10k_dir = '/media/atr17/HDD Storage/Datasets_Download/Got10k'
        # self.trackingnet_dir = '/media/atr17/HDD Storage/Datasets_Download/TrackingNet'
        # self.coco_dir = '/media/atr17/HDD Storage/Datasets_Download/Coco17'
        
        #self.catheter_tracking_dir = '/media/atr17/HDD Storage/Datasets_Download/Catheter_Detection/Images'
        #self.catheter_tracking_dir = '/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Final_Dataset_Rot/Images'
        # self.catheter_tracking_dir = '/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Final_Dataset_Rot/Images'
        #self.catheter_tracking_dir = '/media/atr17/HDD Storage/Datasets_Download/Catheter_Detection/Images/Train'

        # self.catheter_segmentation_dir = '/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset'
        #self.catheter_segmentation_dir = '/media/atr17/HDD Storage/Datasets_Download/Catheter_Detection'
        #self.catheter_segmentation_dir = '/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/new_axial_dataset'
        # self.catheter_segmentation_dir = '/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/new_axial_dataset_dilated2'
        # self.catheter_segmentation_dir = DATASET_ROOT + '/new_axial_dataset_dilated2'
        self.catheter_segmentation_dir = '/home/liming/Documents/new_axial_dataset_dilated2'
        self.catheter_transverse_segmentation_dir = DATASET_ROOT + '/us_simulation3_cactuss'
