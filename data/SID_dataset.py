import torch.utils.data as data
import os
from data.util import glob_file_list,augment_torch_,read_img_path,random_crop_torch

class SIDDatasetFromFolder(data.Dataset):
    def __init__(self,dataset_opt):
        super(SIDDatasetFromFolder, self).__init__()

        data_dir = dataset_opt['data_dir']
        fold_name = dataset_opt.get('fold_name')
        self.crop_size = dataset_opt.get('crop_size')

        self.phase = dataset_opt.get('phase',None)
        self.setsize = dataset_opt.get('setsize')
        self.filp = dataset_opt.get('filp',False)

        lq_data_dir = os.path.join(data_dir, 'short')
        hq_data_dir = os.path.join(data_dir, 'long')

        self.HQ_root, self.LQ_root = hq_data_dir, lq_data_dir

        subfolders_LQ_origin = glob_file_list(self.LQ_root)
        subfolders_GT_origin = glob_file_list(self.HQ_root)

        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': []}

        self.imgs_LQ, self.imgs_GT = {}, {}
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ_origin, subfolders_GT_origin):
            #
            subfolder_name = os.path.basename(subfolder_LQ)

            img_paths_LQ = glob_file_list(subfolder_LQ)
            img_paths_GT = glob_file_list(subfolder_GT)

            max_idx = len(img_paths_LQ)

            self.data_info['path_LQ'].extend(
                img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)

            self.imgs_LQ[subfolder_name] = img_paths_LQ
            self.imgs_GT[subfolder_name] = img_paths_GT
            max_idx = len(img_paths_LQ)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)

        img_LQ_path = self.imgs_LQ[folder][idx]
        img_HQ_path = self.imgs_GT[folder][0]
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
            'folder': folder,
            'idx': self.data_info['idx'][index],
        }
    def __len__(self):
        return len(self.data_info['path_LQ'])

class SID_PairedImage(data.Dataset):
    def __init__(self,dataset_opt):
        super(SID_PairedImage, self).__init__()

        data_dir = dataset_opt['data_dir']
        fold_name = dataset_opt.get('fold_name')
        self.crop_size = dataset_opt.get('crop_size')

        self.phase = dataset_opt.get('phase',None)
        self.setsize = dataset_opt.get('setsize')
        self.filp = dataset_opt.get('filp',False)

        lq_data_dir = os.path.join(data_dir, 'short')
        hq_data_dir = os.path.join(data_dir, 'long')

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



