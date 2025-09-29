import torch.utils.data as data
import os
from data.util import glob_file_list,augment_torch_,read_img_path,random_crop_torch

class UnpairedFromFolder(data.Dataset):
    def __init__(self,dataset_opt):
        super(UnpairedFromFolder, self).__init__()

        data_dir = dataset_opt['data_dir']
        fold_name = dataset_opt.get('fold_name')
        self.crop_size = dataset_opt.get('crop_size')

        self.phase = dataset_opt.get('phase',None)
        self.setsize = dataset_opt.get('setsize')
        self.filp = dataset_opt.get('filp',False)



        subfolders_LQ_origin = glob_file_list(data_dir)


        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': []}

        self.imgs_LQ, self.imgs_GT = {}, {}
        for subfolder_LQ in subfolders_LQ_origin:
            #
            subfolder_name = os.path.basename(subfolder_LQ)

            img_paths_LQ = glob_file_list(subfolder_LQ)

            max_idx = len(img_paths_LQ)

            self.data_info['path_LQ'].extend(
                img_paths_LQ)  # list of path str of images
            self.data_info['folder'].extend([subfolder_name] * max_idx)

            self.imgs_LQ[subfolder_name] = img_paths_LQ
            max_idx = len(img_paths_LQ)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)

        img_LQ_path = self.imgs_LQ[folder][idx]
        #setsize
        img_LQ = read_img_path(img_LQ_path, self.setsize)

        return {
            'lq': img_LQ,
            'lq_path': img_LQ_path,
            'folder': folder,
            'idx': self.data_info['idx'][index],
        }
    def __len__(self):
        return len(self.data_info['path_LQ'])


