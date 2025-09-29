from arch.TPCNet_arch import TPCNet
from arch.scheduler import *
from loss.loss_create import init_loss
from copy import deepcopy
import torch
import os.path as osp
from torchvision import transforms
from metrics.measure import metrics_folder
from arch import define_network
try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False

import os
class BaseModel():
    """Base model."""

    def __init__(self, opt):
        '''

        :param opt:
        opt: opt['train']
        opt_root: opt
        '''
        self.opt_root = opt

        self.opt = opt['train']
        self.device = torch.device('cuda' if self.opt['num_gpu'] != 0 else 'cpu')

        if opt.get('network',None) is not None:
            self.net = define_network(deepcopy(opt['network'])).to(self.device)
        else:
            self.net = TPCNet(CAM_type='YCBCR').to(self.device)
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)

        self.Idx = self.opt.get('Idx',False)
        self.saveGT = self.opt.get('saveGT',False)
        self.target_path = self.opt.get('target_path',None)

        self.opt_eval = self.opt_root.get('eval',None)

        self.setup_optimizers()
        self.setup_schedulers()
        self.loss_fns = init_loss(self.opt['loss_fn'])
        self.loss_value = 0
        self.opt_step_num = 0
        self.net.train()


    def setup_optimizers(self):
        """Set up optimizers."""
        optim_type = self.opt['optim_opt'].pop('type')
        if optim_type == 'Adam':
            self.optimizers = torch.optim.Adam(
                self.net.parameters(), **self.opt['optim_opt'])
        elif optim_type == 'AdamW':
            self.optimizers = torch.optim.AdamW(
                self.net.parameters(), **self.opt['optim_opt'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_type = self.opt['scheduler'].pop('type')
        if scheduler_type in ['CosineAnnealingRestartLR']:
            if self.opt['warmup_epoch'] !=-1:
                scheduler_step = CosineAnnealingRestartLR(optimizer=self.optimizers,
                                                                periods=[self.opt['total_epoch'] - self.opt['warmup_epoch']],
                                                          **self.opt['scheduler'])
                scheduler = GradualWarmupScheduler(self.optimizers, multiplier=1, total_epoch=self.opt['warmup_epoch'],
                                                   after_scheduler=scheduler_step)
                self.schedulers = scheduler
            else:
                scheduler_step = CosineAnnealingRestartLR(optimizer=self.optimizers,
                                                          periods=[self.opt['total_epoch']],
                                                         **self.opt['scheduler']
                                                          )
                self.schedulers = scheduler_step

        elif scheduler_type in ['CosineAnnealingRestartCyclicLR']:
            if self.opt['warmup_epoch'] !=-1:
                periods = self.opt['scheduler'].pop('periods')

                assert sum(periods)+self.opt['warmup_epoch'] == self.opt['total_epoch'],"periods + warmup_epoch not equal total_epoch！"

                periods_ = []
                periods_.append(periods[0]-self.opt['warmup_epoch'])
                periods_.extend(periods[1:])


                scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=self.optimizers,
                                                                periods=periods_,
                                                          **self.opt['scheduler'])
                scheduler = GradualWarmupScheduler(self.optimizers, multiplier=1, total_epoch=self.opt['warmup_epoch'],
                                                   after_scheduler=scheduler_step)
                self.schedulers = scheduler
            else:
                periods = self.opt['scheduler'].pop('periods')
                scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=self.optimizers,
                                                          periods=periods,
                                                          **self.opt['scheduler']
                                                          )
                self.schedulers = scheduler_step

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self):

        self.optimizers.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred =self.net(self.lq)
            self.output = pred
            # loss function
            L1_loss, P_loss, E_loss, D_loss = self.loss_fns
            P_loss = P_loss.to(self.device)
            pred_c = self.net.trans.color_transform(pred)
            gt_c = self.net.trans.color_transform(self.gt)
            loss = L1_loss(pred,self.gt)+self.opt['loss_fn']['P_weight']*P_loss(pred,self.gt)[0]+E_loss(pred,self.gt)+D_loss(pred,self.gt)
            loss_c = L1_loss(pred_c,gt_c)+self.opt['loss_fn']['P_weight']*P_loss(pred_c,gt_c)[0]+E_loss(pred_c,gt_c)+D_loss(pred_c,gt_c)
            loss = loss + loss_c * self.opt['loss_fn']['Color']

        self.amp_scaler.scale(loss).backward()  #防止下溢为0
        self.amp_scaler.unscale_(self.optimizers)  # 在梯度裁剪前先unscale梯度

        if self.opt['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)

        self.amp_scaler.step(self.optimizers)
        self.amp_scaler.update()

        # loss.backward()
        # self.optimizers.step()

        self.loss_value += loss.item()
        self.opt_step_num += 1


    def nonpad_test(self, img=None,alpha_color=None):
        if img is None:
            img = self.lq
        self.net.eval()
        with torch.no_grad():
            if self.opt_eval is None:
                pred = self.net(img)
            else:
                if alpha_color is None:
                    alpha_color = self.opt_eval['color_transition'].get('alpha_color',None)
                if alpha_color is None:
                    pred = self.net(img)
                else:
                    alpha_color_ini = self.net.trans.alpha
                    self.net.trans.alpha=alpha_color
                    self.net.trans.gated2 = True
                    pred = self.net(img)
                    self.net.trans.alpha = alpha_color_ini
                    self.net.trans.gated2 = False
        self.output = pred
        self.net.train()

    def validation(self, eval_dateloder,model_out_path):

        torch.set_grad_enabled(False)
        self.net.load_state_dict(torch.load(model_out_path, map_location=lambda storage, loc: storage))
        for idx, val_data in enumerate(eval_dateloder):
            lq_name = osp.basename(val_data['lq_path'][0])
            gt_name = osp.basename(val_data['gt_path'][0])
            if val_data.get('folder', None) is not None:
                folder = val_data.get('folder', None)[0]
            else:
                folder = None
            self.feed_data(val_data)
            self.nonpad_test()
            output = self.output

            output = torch.clamp(output, 0, 1).to('cpu')
            output_img_p = transforms.ToPILImage()(output.squeeze(0))

            output_img_gt = transforms.ToPILImage()(self.gt.squeeze(0).to('cpu'))
            if self.saveGT or self.target_path is None:
                if self.Idx:
                    output_img_p.save(os.path.join(self.opt_root['path']['img_path_pr'], f'{idx}_'+lq_name))
                    output_img_gt.save(os.path.join(self.opt_root['path']['img_path_pr'], f'{idx}_'+gt_name))
                else:
                    if folder is not None:
                        if not os.path.exists(os.path.join(self.opt_root['path']['img_path_pr'],folder)):
                            os.makedirs(os.path.join(self.opt_root['path']['img_path_pr'],folder))
                        output_img_p.save(os.path.join(self.opt_root['path']['img_path_pr'],folder,lq_name))
                        output_img_gt.save(os.path.join(self.opt_root['path']['img_path_pr'],folder,gt_name))
            elif self.target_path is not None:
                if self.Idx:
                    output_img_p.save(os.path.join(self.opt_root['path']['img_path_pr'],f'{idx}_'+lq_name))
                    # output_img_gt.save(self.opt_root['path']['img_path_gt']+'/'+f'{idx}_'+gt_name+'.png')
                else:
                    if folder is not None:
                        if not os.path.exists(os.path.join(self.opt_root['path']['img_path_pr'],folder)):
                            os.makedirs(os.path.join(self.opt_root['path']['img_path_pr'],folder))
                        output_img_p.save(os.path.join(self.opt_root['path']['img_path_pr'],folder,lq_name))
                    else:
                        output_img_p.save(os.path.join(self.opt_root['path']['img_path_pr'],lq_name))
                    # output_img_gt.save(self.opt_root['path']['img_path_gt']+'/'+gt_name+'.png')
            else:
                if self.Idx:
                    output_img_p.save(os.path.join(self.opt_root['path']['img_path_pr'],f'{idx}_'+lq_name))
                    # output_img_gt.save(self.opt_root['path']['img_path_gt']+'/'+f'{idx}_'+gt_name+'.png')
                else:
                    if folder is not None:
                        if not os.path.exists(os.path.join(self.opt_root['path']['img_path_pr'],folder)):
                            os.makedirs(os.path.join(self.opt_root['path']['img_path_pr'],folder))
                        output_img_p.save(os.path.join(self.opt_root['path']['img_path_pr'],folder,lq_name))
                    else:
                        output_img_p.save(os.path.join(self.opt_root['path']['img_path_pr'],lq_name))
                    # output_img_gt.save(self.opt_root['path']['img_path_gt']+'/'+gt_name+'.png')
            del self.gt
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
        # self.saveGT = False # GT only save a time
        torch.set_grad_enabled(True)
    def get_epoch_loss(self):
        if self.opt_step_num==0:
            epoch_loss = self.loss_value
            self.loss_value = 0
            self.opt_step_num=0
            return epoch_loss,self.optimizers.param_groups[0]['lr']
        else:
            epoch_loss = self.loss_value / self.opt_step_num
            self.loss_value = 0
            self.opt_step_num=0
            return epoch_loss,self.optimizers.param_groups[0]['lr']


    def visualization(self):
        img_out = transforms.ToPILImage()((self.output[0].squeeze(0).cpu()))
        gt_out =  transforms.ToPILImage()((self.gt[0].squeeze(0).cpu()))
        img_out.save(self.opt_root['path']['experiments_root'] + '/' + 'test.png')
        gt_out.save(self.opt_root['path']['experiments_root'] + '/' + 'gt.png')



    def evaluation(self,epoch_current,iter_current,epoch_loss,use_GT_mean=False):

        if self.target_path is None:
            avg_psnr, avg_ssim, avg_lpips = metrics_folder(self.opt_root['path']['img_path_pr'], self.opt_root['path']['img_path_gt'], use_GT_mean)
        else:
            avg_psnr, avg_ssim, avg_lpips = metrics_folder(self.opt_root['path']['img_path_pr'],self.target_path, use_GT_mean)

        log_vars = {'epoch': epoch_current, 'iter': iter_current}
        log_vars.update(
            {'avg_psnr': avg_psnr, 'avg_ssim': avg_ssim, 'avg_lpips': avg_lpips, 'epoch_loss': epoch_loss})
        log_lrs = {'lrs': self.optimizers.param_groups[0]['lr']}
        log_vars.update(log_lrs)
        return log_vars






    def save_train_state(self,epoch_current, iter_current):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if iter_current != -1:
            state = {
                'epoch': epoch_current,
                'iter': iter_current,
                'optimizers': [],
                'schedulers': []
            }

            state['optimizers'].append(self.optimizers.state_dict())

            state['schedulers'].append(self.schedulers.state_dict())

            # best metric
            save_filename = f'{iter_current}.state'
            save_path = os.path.join(self.opt_root['path']['training_states'],
                                     save_filename)
            torch.save(state, save_path)


    def save_network(self,epoch_current):
        '''
        save network
        '''
        model_out_path = self.opt_root['path']['models'] + '/' + "epoch_{}.pth".format(epoch_current)
        torch.save(self.net.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        return model_out_path

    def load_network(self,net_path):
        '''
        :param net_path: .pth network weight
        :return:
        '''
        modle_pth = torch.load(net_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(modle_pth)



    def resume_train(self,state_resume=None):
        '''
        default: resume the lastest state path
        '''
        if state_resume is None:
            state_folder_path = self.opt_root['path']['training_states']
            state_paths = os.listdir(state_folder_path)
            if len(state_paths) > 0:

                state_max_name = max([int(x[:-6]) for x in state_paths])
                state_resume_name = '{}.state'.format(state_max_name)
                state_resume = os.path.join(state_folder_path, state_resume_name)
                state = torch.load(state_resume, map_location=lambda storage, loc: storage)

                epoch_resume = state['epoch']
                iter_resume = state['iter']
                max_model_name = 'epoch_{}.pth'.format(epoch_resume)
                model_resume = os.path.join(self.opt_root['path']['models'], max_model_name)
                modle_pth = torch.load(model_resume, map_location=lambda storage, loc: storage)

                self.net.load_state_dict(modle_pth)
                self.optimizers.load_state_dict(state['optimizers'][0])
                self.schedulers.load_state_dict(state['schedulers'][0])


                epoch_start = epoch_resume
                iter_start = iter_resume

            else:
                epoch_start = 0
                iter_start = 0




            return epoch_start,iter_start
        else:
            state = torch.load(state_resume, map_location=lambda storage, loc: storage)
            self.optimizers.load_state_dict(state['optimizers'][0])
            self.schedulers.load_state_dict(state['schedulers'][0])
            modle_pth = os.path.join(self.opt_root['path']['models'],f"epoch_{state['epoch']}.pth")
            self.net.load_state_dict(torch.load(modle_pth, map_location=lambda storage, loc: storage))

            epoch_start = state['epoch']
            iter_start = state['iter']
            return epoch_start,iter_start































