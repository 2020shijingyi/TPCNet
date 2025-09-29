from data.SID_dataset import *
from data.util import *
import importlib
from os import path as osp
from torch.utils.data import DataLoader

__all__ = ['create_dataset']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in os.listdir(data_folder)
    if v.endswith('_dataset.py')
]
# stx()
# import all the dataset modules，所有结尾带_dataset都被import成包了
_dataset_modules = [
    importlib.import_module(f'data.{file_name}')
    for file_name in dataset_filenames
]


# stx()
def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt.pop('type')

    # dynamic instantiation  如何理解动态实例化？逐个遍历已经import进来的包，就是所有dataset.py
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)  #getattr() 获取对象的属性值
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    return dataset

def create_dataloader(dataset,loader_opt):

    return DataLoader(dataset,**loader_opt)





