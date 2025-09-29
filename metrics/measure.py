import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
import pyiqa
def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))
    return psnr

def has_file_types(folder_path, exts=('JPG','PNG','jpg','png', 'bmp', 'tiff', 'tif')):
    return any(glob.glob(os.path.join(folder_path, f"*.{ext}")) for ext in exts)

def metrics_folder(pr_paths, gt_paths, use_GT_mean):

    if has_file_types(pr_paths):
        ext = os.path.splitext(pr_paths[0])[1]  # 包含点 .jpg
        pr_lists = sorted(glob.glob(os.path.join(pr_paths, '*'+ext)))
        pr_folder = False
    else:
        pr_lists = sorted(glob.glob(os.path.join(pr_paths,'*')))
        pr_folder = True
    if has_file_types(gt_paths):
        ext = os.path.splitext(gt_paths[0])[1]  # 包含点 .jpg
        gt_lists = sorted(glob.glob(os.path.join(gt_paths, '*'+ext)))
        gt_folder = False
    else:
        gt_lists = sorted(glob.glob(os.path.join(gt_paths,'*')))
        gt_folder = True
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    n=0
    for pr_folder_paths, gt_folder_paths in tqdm(zip(pr_lists, gt_lists), desc='materics', total=len(pr_lists),
                                      ncols=100, unit='num'):
        if gt_folder:
            gt_paths = sorted(glob.glob(os.path.join(gt_folder_paths,'*')))
        else:
            gt_paths = [gt_folder_paths]
        if pr_folder:
            pr_paths = sorted(glob.glob(os.path.join(pr_folder_paths, '*')))
        else:
            pr_paths = [pr_folder_paths]
        for i,pr_path in enumerate(pr_paths):
            n = n + 1

            im1 = Image.open(pr_paths[i]).convert('RGB')
            im2 = Image.open(gt_paths[0]).convert('RGB')

            (h, w) = im2.size
            im1 = im1.resize((h, w))
            im1 = np.array(im1)
            im2 = np.array(im2)

            if use_GT_mean:
                mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
                mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
                im1 = np.clip(im1 * (mean_target / mean_restored), 0, 255)

            score_psnr = calculate_psnr(im1, im2)
            score_ssim = calculate_ssim(im1, im2)
            ex_p0 = lpips.im2tensor(im1).cuda()
            ex_ref = lpips.im2tensor(im2).cuda()

            score_lpips = loss_fn.forward(ex_ref, ex_p0)

            avg_psnr += score_psnr
            avg_ssim += score_ssim
            avg_lpips += score_lpips.item()
            torch.cuda.empty_cache()

    avg_psnr = avg_psnr / n
    avg_ssim = avg_ssim / n
    avg_lpips = avg_lpips / n
    return avg_psnr, avg_ssim, avg_lpips

def metrics_unpaired(pr_paths,TotalAvg = False,metrics = ['niqe']):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## Initialization metrics & fn
    metrics_record = {}
    metrics_fn = {}
    for metric in metrics:
        metrics_record['avg_' + metric] = 0
        metrics_fn[metric + '_fn'] = pyiqa.create_metric(metric, device=device)
        if TotalAvg:
            metrics_record['total_'+metric] = 0
    ##
    if has_file_types(pr_paths):
        ext = os.path.splitext(pr_paths[0])[1]  # include .jpg
        pr_lists = sorted(glob.glob(os.path.join(pr_paths, '*'+ext)))
        pr_folder = False
    else:
        pr_lists = sorted(glob.glob(os.path.join(pr_paths,'*')))
        pr_folder = True

    n = 0
    materials = {}
    for k,pr_folder_paths in tqdm(enumerate(pr_lists), desc='materics', total=len(pr_lists),ncols=100, unit='num'):
        if pr_folder:
            pr_paths = sorted(glob.glob(os.path.join(pr_folder_paths, '*')))
        else:
            pr_paths = [pr_folder_paths]

        folder = pr_folder_paths.split(os.sep)[-1]
        for i,pr_path in enumerate(pr_paths):
            n = n + 1
            for (fn_name,fn) in metrics_fn.items():
                score = fn(pr_paths[i]).item()
                metric = fn_name.split('_')[-2]
                metrics_record['avg_' + metric] +=score
            torch.cuda.empty_cache()

        # calculate each avg metrics for a dataset e.g DICM ... NPE
        for record_name,value in metrics_record.items():
            metric = record_name.split('_')[-1]
            if 'avg' in record_name:
                materials[folder + '_'+metric] = value / n
                metrics_record[record_name] = value / n


        if TotalAvg:
            for record_name, value in metrics_record.items():
                metric = record_name.split('_')[-1]
                if 'total' in record_name:
                    metrics_record[record_name] += metrics_record['avg_' + metric]

        # clear
        n=0
        for record_name, value in metrics_record.items():
            if 'avg' in record_name:
                metrics_record[record_name]=0
    if TotalAvg:
        for record_name, value in metrics_record.items():
            if 'total' in record_name:
                materials[record_name]= metrics_record[record_name] / (k+1)
    return materials

