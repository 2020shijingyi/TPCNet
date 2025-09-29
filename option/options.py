import yaml
from collections import OrderedDict
from os import path as osp
import os
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_create(yaml_path =''):
    with open(yaml_path, mode='r', encoding='utf-8') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
        opt['path'] = {}
        # for key, val in opt['path'].items():
        #     if (val is not None) and ('resume_state' in key
        #                               or 'pretrain_network' in key):
        #         opt['path'][key] = osp.expanduser(val)
        opt['path']['root'] = osp.abspath(
            osp.join(__file__, osp.pardir, osp.pardir))
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')
        opt['path']['img_path_gt'] = os.path.join(opt['path']['visualization'], opt['dataname'], 'gt')
        opt['path']['img_path_pr'] = os.path.join(opt['path']['visualization'], opt['dataname'], 'pr')

        path_opt = opt['path'].copy()
        for key, path in path_opt.items():
            if ('strict_load' not in key) and ('pretrain_network'
                                               not in key) and ('resume'
                                                                not in key):
                os.makedirs(path, exist_ok=True)
    return opt