from option.options import opt_create
from data.dataset_create import DatasetCreate
from arch.model_create import BaseModel
from logger.loging import init_loggers
from logger.logger import MessageLogger,get_root_logger
import random
import numpy as np
import torch
import os
from tqdm import tqdm
def seed_torch():
    # seed = random.randint(1, 1000000)
    seed = 114
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(yaml_path=''):
    # set seed
    seed_torch()
    # load option
    opt = opt_create(yaml_path=yaml_path)
    logger, tb_logger = init_loggers(opt)
    # load data
    # train & eval
    training_data_loader, eval_data_loader = DatasetCreate(opt['datasets'])
    opt['traindata_num'] = len(training_data_loader)

    model = BaseModel(opt=opt)

    # crop_size = opt['train'].get('crop_size')
    # img_size = opt['train'].get('img_size')
    # resume
    if opt['resume']:
        # epoch_start,iter_start = model.resume_train('/root/autodl-tmp/TPCNet_new/experiments/TPCNet_LOLV2_S_4/training_states/54240.state')
        epoch_start, iter_start = model.resume_train()
    else:
        epoch_start = 0
        iter_start = 0

    msg_logger = MessageLogger(opt, epoch_start, tb_logger)

    iter_current = iter_start
    for epoch_current in range(epoch_start + 1, opt['train']['total_epoch'] + 1):
        for train_data in tqdm(training_data_loader):
            iter_current = iter_current + 1
            lq = train_data['lq']
            gt = train_data['gt']
            model.feed_train_data({'lq': lq, 'gt': gt})
            model.optimize_parameters()
        epoch_loss, lrs = model.get_epoch_loss()
        print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch_current, epoch_loss, lrs))
        model.schedulers.step(epoch_current)
        model.visualization()
        if epoch_current % opt['logger']['save_freq'] == 0:
            model_out_path = model.save_network(epoch_current)
            model.save_train_state(epoch_current, iter_current)

            model.validation(eval_data_loader, model_out_path)
            log_vars = model.evaluation(epoch_current, iter_current, epoch_loss)
            msg_logger(log_vars)




if __name__ == '__main__':
    main('E:\Open_source\TPCNet_new\option\TPCNet_SID.yml')

























