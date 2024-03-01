from lib.test.evaluation.environment import EnvSettings

# PATH = "/mnt/c/Users/steve/OneDrive/Documents/12SemesterTUM/IDP/repos/AiAProj"
PATH = "/media/liming/Data/IDP/AiAProj"
DATASET_ROOT = "/media/liming/Data/IDP/dataset"
def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here

    settings.got10k_path = PATH + '/GOT10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    #settings.lasot_path = PATH + '/LaSOT'
    settings.lasot_path = '/media/atr17/HDD Storage/Datasets_Download/LaSOT/LaSOT'
    # Set your image path for the catheter
    #settings.catheter_path = '/media/atr17/HDD Storage/Datasets_Download/Catheter_Detection/Images'
    # settings.catheter_path = '/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Final_Dataset_Rot/Images'
    # settings.catheter_path = PATH + '/new_axial_mini/Images/'
    # settings.catheter_path = '/mnt/c/Users/steve/OneDrive/Documents/12SemesterTUM/IDP/repos/dataset/new_axial_dataset_dilated2/Images'
    settings.catheter_path = '/media/liming/Data/IDP/dataset/new_axial_dataset_dilated2/Images'
    #settings.catheterseg_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset"
    # settings.catheterseg_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/new_axial_dataset_dilated2"
    # settings.catheterseg_path = PATH + '/new_axial_mini'
    # settings.catheterseg_path = '/home/liming/Documents/dataset/new_axial_dataset_dilated2'
    # settings.catheterseg_path = '/mnt/c/Users/steve/OneDrive/Documents/12SemesterTUM/IDP/repos/dataset/new_axial_dataset_dilated2'
    # settings.catheterseg_path = '/media/liming/Data/IDP/dataset/new_axial_dataset_dilated2'
    settings.catheterseg_path= '/home/liming/Documents/new_axial_dataset_dilated2'
    settings.cathetertransseg_path = DATASET_ROOT + '/us_simulation3_cactuss'
    # settings.cathetertransseg_path = DATASET_ROOT + '/us_phantom'

    # settings.network_path = PATH + '/AiAReSeg/test/networks'  # Where tracking networks are stored
    settings.network_path = PATH + '/AiAReSeg/pretrained_networks'  # Where tracking networks are stored
    settings.nfs_path = PATH + '/NFS30'
    settings.otb_path = PATH + '/OTB100'
    settings.prj_dir = PATH + '/AiAReSeg'
    settings.result_plot_path = PATH + '/AiAReSeg/test/result_plots'
    settings.results_path = PATH + '/AiAReSeg/test/tracking_results'  # Where to store tracking results
    settings.save_dir = PATH + '/AiAReSeg'
    settings.segmentation_path = PATH + '/AiAReSeg/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.trackingnet_path = PATH + '/TrackingNet'
    settings.uav_path = PATH + '/UAV123'
    settings.show_result = False

    return settings

