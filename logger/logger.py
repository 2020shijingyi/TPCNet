import datetime
import logging
import time
from logger.dist_util import get_dist_info, master_only



initialized_logger = {}


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_epochs=1, tb_logger=None):
        self.exp_name = opt['name']
        self.start_epochs = start_epochs
        self.max_epoch = opt['train']['total_epoch']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()
        self.csv_logger = get_root_logger(logger_name='metric')
        self.csv_logger.info('avg_psnr,avg_ssim,avg_lpips,epoch')

    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lr = log_vars.pop('lrs')

        message = f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, ' f'iter:{current_iter:8,d}, lr:('

        message += f'{lr:.3e},'
        message += ')] '

        # time and estimated time

        total_time = time.time() - self.start_time
        time_sec_avg = total_time / (epoch - self.start_epochs + 1)
        eta_sec = time_sec_avg * (self.max_epoch - epoch - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        message += f'[eta: {eta_str}]'
        # other items, especially losses
        epoch_loss = log_vars.pop('epoch_loss')
        message += f' [epoch_loss:{epoch_loss}]'
        self.tb_logger.add_scalar(f'epoch_loss',epoch_loss,epoch)

        for k, v in log_vars.items():
            #message += f'{k}: {v:.4e} '
            # tensorboard logger
            if self.use_tb_logger:
                self.tb_logger.add_scalar(k, v, epoch)
        self.logger.info(message)
        self.logger.info('')
        materials = 'Validation ValSet   #'
        for k, v in log_vars.items():
            materials += f'{k}: {v:.4f} '
        csv_maertial = ','.join(f"{v:.4f}" for v in log_vars.values())+',{}'.format(str(epoch))
        self.csv_logger.info(csv_maertial)  # 数据
        self.logger.info(materials)




@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = logging.getLogger('basicsr')

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(id=wandb_id, resume=resume,
               name=opt['name'], config=opt, project=project, sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger
    if logger_name != 'metric':
        format_str = '%(asctime)s %(levelname)s: %(message)s'
    else:
        format_str = ''
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        if logger_name != 'metric':
            file_handler = logging.FileHandler(log_file, 'w')
        else:
            file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger



