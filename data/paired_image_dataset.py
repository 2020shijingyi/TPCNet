from torch.utils import data as data
import os
from data.util import glob_file_list,augment_torch_,read_img_path,random_crop_torch

class Dataset_PairedImage(data.Dataset):
    def __init__(self,dataset_opt):
        super(Dataset_PairedImage, self).__init__()

        data_dir = dataset_opt['data_dir']
        fold_name = dataset_opt.get('fold_name')
        self.crop_size = dataset_opt.get('crop_size')

        self.phase = dataset_opt.get('phase',None)
        self.setsize = dataset_opt.get('setsize')
        self.filp = dataset_opt.get('filp',False)

        if fold_name is None:
            lq_data_dir = os.path.join(data_dir,'Low')
            hq_data_dir = os.path.join(data_dir, 'Normal')
        else:
            lq_data_dir = os.path.join(data_dir,fold_name)
            hq_data_dir = os.path.join(data_dir, fold_name)

        self.HQ_root, self.LQ_root = hq_data_dir, lq_data_dir


        LQ_origin = glob_file_list(self.LQ_root)
        HQ_origin = glob_file_list(self.HQ_root)


        self.LQ_origin_list = LQ_origin
        self.HQ_origin_list = HQ_origin

    def __getitem__(self, index):
        img_LQ_path = self.LQ_origin_list[index]
        img_HQ_path = self.HQ_origin_list[index]
        #setsize
        if self.phase == 'train':
            img_LQ = read_img_path(img_LQ_path, self.setsize)
            img_HQ = read_img_path(img_HQ_path, self.setsize)

            if self.crop_size is not None:
                img_LQ,img_HQ = random_crop_torch(img_LQ,img_HQ,self.crop_size)

            img_LQ,img_HQ = augment_torch_(img_LQ,img_HQ, self.filp, self.filp)# data augment
        else:
            img_LQ = read_img_path(img_LQ_path, self.setsize)
            img_HQ = read_img_path(img_HQ_path, self.setsize)

        return {
            'lq': img_LQ,
            'gt': img_HQ,
            'lq_path': img_LQ_path,
            'gt_path': img_HQ_path,
        }
    def __len__(self):
        return len(self.LQ_origin_list)
class LOLV1_Dataset_PairedImage(data.Dataset):
    def __init__(self,dataset_opt):
        super(LOLV1_Dataset_PairedImage, self).__init__()

        data_dir = dataset_opt['data_dir']
        fold_name = dataset_opt.get('fold_name')
        self.crop_size = dataset_opt.get('crop_size')

        self.phase = dataset_opt.get('phase',None)
        self.setsize = dataset_opt.get('setsize')
        self.filp = dataset_opt.get('filp',False)

        if fold_name is None:
            lq_data_dir = os.path.join(data_dir,'input')
            hq_data_dir = os.path.join(data_dir, 'target')
        else:
            lq_data_dir = os.path.join(data_dir,fold_name)
            hq_data_dir = os.path.join(data_dir, fold_name)

        self.HQ_root, self.LQ_root = hq_data_dir, lq_data_dir


        LQ_origin = glob_file_list(self.LQ_root)
        HQ_origin = glob_file_list(self.HQ_root)


        self.LQ_origin_list = LQ_origin
        self.HQ_origin_list = HQ_origin

    def __getitem__(self, index):
        img_LQ_path = self.LQ_origin_list[index]
        img_HQ_path = self.HQ_origin_list[index]
        #setsize
        if self.phase == 'train':
            img_LQ = read_img_path(img_LQ_path, self.setsize)
            img_HQ = read_img_path(img_HQ_path, self.setsize)

            if self.crop_size is not None:
                img_LQ,img_HQ = random_crop_torch(img_LQ,img_HQ,self.crop_size)

            img_LQ,img_HQ = augment_torch_(img_LQ,img_HQ, self.filp, self.filp)# data augment
        else:
            img_LQ = read_img_path(img_LQ_path, self.setsize)
            img_HQ = read_img_path(img_HQ_path, self.setsize)

        return {
            'lq': img_LQ,
            'gt': img_HQ,
            'lq_path': img_LQ_path,
            'gt_path': img_HQ_path,
        }
    def __len__(self):
        return len(self.LQ_origin_list)

