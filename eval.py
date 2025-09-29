from option.options import opt_create
from data.dataset_create import DatasetCreate
from arch.model_create import BaseModel
from metrics.measure import metrics_folder,metrics_unpaired
from torchvision import transforms
import random
import numpy as np
import torch
import os
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F

def pad_to_even(img):

    if img.dim() == 3:
        _, h, w = img.shape
    elif img.dim() == 4:
        _, _, h, w = img.shape
    else:
        raise ValueError("input shape must be [C,H,W] or [B,C,H,W]")

    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8

    pad_h = new_h - h
    pad_w = new_w - w

    # pad格式：(left, right, top, bottom)
    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)

    return img_pad, {"pad_h": pad_h, "pad_w": pad_w}
def crop_to_original(img, pad_info):
    '''
    img shape b c h w or c h w

    '''

    if pad_info is not None:
        pad_h = pad_info["pad_h"]
        pad_w = pad_info["pad_w"]

        if pad_h > 0:
            img = img[..., :-pad_h, :]
        if pad_w > 0:
            img = img[..., :, :-pad_w]
        return img
    else:
        return img


def seed_torch():
    # seed = random.randint(1, 1000000)
    seed = 114
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def eval_and_reconstruction(yaml_path,weight_path,save_paths,target_path=None,alpha_color=None,use_GT_mean=False,Unpaired = False):
    seed_torch()
    opt = opt_create(yaml_path)
    weght_path = weight_path
    results_save_path  = save_paths
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    target_path = target_path
    # alpha_color = 0.8
    model = BaseModel(opt=opt)
    if len(opt['datasets'])==1:
        eval_data_loader = DatasetCreate(opt['datasets'])
    else:
        _,eval_data_loader = DatasetCreate(opt['datasets'])

    model.load_network(weght_path)
    for idx, val_data in enumerate(tqdm(eval_data_loader)):
        # lq_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
        lq_name = osp.basename(val_data['lq_path'][0])
        # folder = val_data.get('folder', None)[0]
        if val_data.get('folder', None) is not None:
            folder = val_data.get('folder', None)[0]
        else:
            folder = None
        _,_,h,w = val_data['lq'].shape
        if h%8!=0 or w%8!=0:
            lq_pad,pad_info = pad_to_even(val_data['lq'])
            val_data['lq'] = lq_pad
        else:
            pad_info = None
        model.feed_data(val_data)
        model.nonpad_test()
        output = model.output
        output = crop_to_original(output,pad_info)
        output = torch.clamp(output, 0, 1).to('cpu')
        output_img_p = transforms.ToPILImage()(output.squeeze(0))
        if folder is not None:
            if not os.path.exists(os.path.join(results_save_path, folder)):
                os.makedirs(os.path.join(results_save_path, folder))
            output_img_p.save(os.path.join(results_save_path,folder,lq_name))
        else:
            output_img_p.save(os.path.join(results_save_path,lq_name))
        # tentative for out of GPU memory
        del model.lq
        del model.output
        torch.cuda.empty_cache()
    if Unpaired:
        print("===> Eval results <=== ")
    else:
        avg_psnr, avg_ssim, avg_lpips = metrics_folder(results_save_path, target_path,use_GT_mean=use_GT_mean)
        print("===> Eval results <=== ")
        print("===> avg_psnr: {:.4f} || avg_ssim: {:.4f} || avg_lpips: {:.4f} ||".format(avg_psnr, avg_ssim, avg_lpips))

def eval_unpaired(Unpaired_path,TotalAvg = False,metrics = ['niqe','musiq','pi']):
    '''
    metrics is the metric name list
    TotalAvg is for calculating all datasets metrics average value
    '''
    metric = metrics_unpaired(Unpaired_path,TotalAvg,metrics)
    for (dataname, value_mean) in metric.items():
        name = dataname.split('_')[-2]
        mater = dataname.split('_')[-1]
        print("===>Eval " + name + " results<===", "===>" + mater + "<==={:.4f}".format(value_mean))

def eval_wiout_reconstruction(save_paths, target_path,use_GT_mean=False):

    avg_psnr, avg_ssim, avg_lpips = metrics_folder(save_paths, target_path, use_GT_mean=use_GT_mean)
    print("===> Eval results <=== ")
    print("===> avg_psnr: {:.4f} || avg_ssim: {:.4f} || avg_lpips: {:.4f} ||".format(avg_psnr, avg_ssim, avg_lpips))




if __name__ == '__main__':
    ## eval_wiout_reconstruction
    # results = r'E:\TopList\PairLIE\Result\LCDP2\input'
    # target_path = r'F:\LCDP\LCDP\test\gt'
    # eval_wiout_reconstruction(results,target_path)
    ##eval_reconstruction
    results_save_path = r'Results\\LCDP'
    yaml_path = r'option/TPCNet_LCDP.yml'
    weght_path = r"Weight/LCDP.pth"
    target_path = r'F:\LCDP\LCDP\test\gt'
    eval_and_reconstruction(yaml_path,weght_path,save_paths=results_save_path,target_path=target_path,use_GT_mean=False,Unpaired=False)




