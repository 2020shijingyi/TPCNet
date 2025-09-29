import glob
import numpy as np
import torch
import cv2
import os
import random
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def read_img2(path, size=None):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    img = np.load(path)

    if size is not None:
        img = cv2.resize(img, (size[0], size[1]))
            # img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img
def read_img_seq2(path, size=None):
    """Read a sequence of images from a given folder path
    Args:
        path (list/str): list of image paths/image folder path

    Returns:
        imgs (Tensor): size (T, C, H, W), RGB, [0, 1]
    """

    img_l = [read_img2(v, size) for v in path]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    try:
        imgs = imgs[:, :, :, [2, 1, 0]]
    except Exception:
        import ipdb; ipdb.set_trace()
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def read_img_path(path, size=None):
    '''

    read image from jpg;png...
    '''
    img_l = np.array(load_img(path))

    if size is not None:
        img_l = cv2.resize(img_l,size)

    img_l = img_l / 255.0
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_l, (2, 0, 1)))).float()
    return imgs




def augment_torch(img_list, hflip=True, vflip=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    # rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = flip(img, 2)
        if vflip:
            img = flip(img, 1)
        # if rot90:
        #     # import pdb; pdb.set_trace()
        #     img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

def augment_torch_(lq,hq, hflip=True, vflip=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    # rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = torch.flip(img, dims=[2])
        if vflip:
            img = torch.flip(img, dims=[1])
        # if rot90:
        #     # import pdb; pdb.set_trace()
        #     img = img.transpose(1, 0, 2)
        return img

    return _augment(lq),_augment(hq)

def random_crop_torch(img_lqs, img_gts, patch_size):
    """
    Randomly crop paired LQ and GT tensors (torch.Tensor in [C, H, W] format).

    """
    _, h_lq, w_lq = img_lqs.shape
    _, h_gt, w_gt = img_gts.shape

    # Random top-left crop position for LQ
    top = random.randint(0, h_lq - patch_size)
    left = random.randint(0, w_lq - patch_size)

    img_lqs = img_lqs[:, top:top + patch_size, left:left + patch_size]

    # Corresponding top-left for GT
    img_gts = img_gts[:, top:top + patch_size, left:left + patch_size]

    return img_lqs, img_gts


# pr_paths = ''
# pr_list = os.path.join(pr_paths,'*.png')