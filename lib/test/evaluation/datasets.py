import importlib
from collections import namedtuple

from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = 'lib.test.evaluation.%sdataset'  # Useful abbreviations to reduce the clutter

# Expand this dictionary to include the catheter tracking dataset
dataset_dict = dict(
    otb=DatasetInfo(module=pt % 'otb', class_name='OTBDataset', kwargs=dict()),
    nfs=DatasetInfo(module=pt % 'nfs', class_name='NFSDataset', kwargs=dict()),
    uav=DatasetInfo(module=pt % 'uav', class_name='UAVDataset', kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % 'trackingnet', class_name='TrackingNetDataset', kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % 'got10k', class_name='GOT10KDataset', kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % 'got10k', class_name='GOT10KDataset', kwargs=dict(split='val')),
    lasot=DatasetInfo(module=pt % 'lasot', class_name='LaSOTDataset', kwargs=dict(subset='testing')),
    lasot_ext=DatasetInfo(module=pt % 'lasot', class_name='LaSOTDataset', kwargs=dict(subset='extension')),
    catheter_tracking_test=DatasetInfo(module=pt % 'catheter', class_name='CatheterDataset', kwargs=dict(subset='Val')),
    catheter_segmentation_test=DatasetInfo(module=pt % 'catheterseg', class_name='CatheterSegDataset', kwargs=dict(subset='Val')),
    catheter_transverse_segmentation_test=DatasetInfo(module=pt % 'cathetertransseg', class_name='CatheterTransSegDataset', kwargs=dict(subset='Val'))

)


def load_dataset(name: str):
    """
    Import and load a single dataset.
    """

    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('ERROR: unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """
    Get a single or set of datasets.
    """

    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset
