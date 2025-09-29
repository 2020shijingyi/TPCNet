from data.SID_dataset import SIDDatasetFromFolder
from torch.utils.data import DataLoader
from data import create_dataset,create_dataloader


def DatasetCreate(opt):
    training_data_loader,eval_data_loader = None,None

    for phase, phase_opt in opt.items():
        if phase =='train':
            #train dataset
            phase_opt['dateset_opt']['phase']=phase
            train_set = create_dataset(phase_opt['dateset_opt'])
            training_data_loader = create_dataloader(train_set,phase_opt['loader_opt'])
        else:
            # eval dataset
            phase_opt['dateset_opt']['phase'] = phase
            eval_set = create_dataset(phase_opt['dateset_opt'])
            eval_data_loader = create_dataloader(eval_set,phase_opt['loader_opt'])

    if training_data_loader is None:
        return eval_data_loader
    elif eval_data_loader is None:
        return training_data_loader
    else:
        return training_data_loader,eval_data_loader


