import torch.utils.data as data
import os
from data.util import glob_file_list,augment_torch_,read_img_path,random_crop_torch

class LCDPDataset(data.Dataset):
    def __init__(self,dataset_opt):
        super(LCDPDataset, self).__init__()

        data_dir = dataset_opt['data_dir']
        fold_name = dataset_opt.get('fold_name')
        self.crop_size = dataset_opt.get('crop_size')

        self.phase = dataset_opt.get('phase',None)
        self.setsize = dataset_opt.get('setsize')
        self.filp = dataset_opt.get('filp',False)

        lq_data_dir = os.path.join(data_dir, 'input')
        hq_data_dir = os.path.join(data_dir, 'gt')

        self.HQ_root, self.LQ_root = hq_data_dir, lq_data_dir

        subfolders_LQ_origin = glob_file_list(self.LQ_root)
        subfolders_GT_origin = glob_file_list(self.HQ_root)

        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': []}

        self.imgs_LQ, self.imgs_GT = subfolders_LQ_origin, subfolders_GT_origin

    def __getitem__(self, index):

        img_LQ_path = self.imgs_LQ[index]
        img_HQ_path = self.imgs_GT[index]
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
        return len(self.imgs_LQ)



