"""Command-line + YAML option parsing for train.py (based on BasicSR's parse_options)."""
import os
import random
import argparse
from collections import OrderedDict

import torch
import yaml

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import _postprocess_yml_value


def ordered_yaml_loader():
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, dict_constructor)
    return Loader


def yaml_load(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=ordered_yaml_loader())


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--weights', type=str, default=None,
                        help='Generator checkpoint to start from; overrides path:pretrain_network_g.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--force_yml', nargs='+', default=None,
                        help='Override yml values, e.g. train:ema_decay=0.999 (floats as !!float2e-5)')
    args = parser.parse_args()

    opt = yaml_load(args.opt)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.force_yml is not None:
        for entry in args.force_yml:
            keys, value = entry.split('=')
            node = opt
            keys = keys.strip().split(':')
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = _postprocess_yml_value(value.strip())

    if args.weights:
        opt['path']['pretrain_network_g'] = args.weights

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase.split('_')[0]  # e.g. val_1 -> val
        if 'scale' in opt:
            dataset['scale'] = opt['scale']

    for key, val in opt['path'].items():
        if val is not None and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = os.path.expanduser(val)

    experiments_root = os.path.join(opt['path'].get('experiments_root') or os.path.join(root_path, 'experiments'),
                                    opt['name'])
    opt['path']['experiments_root'] = experiments_root
    opt['path']['models'] = os.path.join(experiments_root, 'models')
    opt['path']['training_states'] = os.path.join(experiments_root, 'training_states')
    opt['path']['log'] = experiments_root
    opt['path']['visualization'] = os.path.join(experiments_root, 'visualization')

    if 'debug' in opt['name']:
        if 'val' in opt:
            opt['val']['val_freq'] = 8
        opt['logger']['print_freq'] = 1
        opt['logger']['save_checkpoint_freq'] = 8

    return opt, args
