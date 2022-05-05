from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

cfg = CN()

cfg.task = 'manhattan_sdf'
cfg.exp_name = 'scannet'
cfg.gpus = [0]
cfg.pretrain = ''
cfg.resume = True
cfg.distributed = False
cfg.fix_random = False

# module
cfg.train_dataset_module = 'lib.datasets.scannet'
cfg.test_dataset_module = 'lib.datasets.scannet'
cfg.network_module = 'lib.neworks.network'
cfg.trainer_module = 'lib.train.trainers.manhattan_sdf'
cfg.evaluator_module = 'lib.evaluators.mesh'

# epoch
cfg.ep_iter = -1
cfg.save_ep = 1
cfg.save_latest_ep = 1
cfg.eval_ep = 1
log_interval: 20

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = 'default'
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0.
cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})
cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.val_dataset = ''
cfg.test.batch_size = 1
cfg.test.collator = 'default'
cfg.test.epoch = -1
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})

# recorder
cfg.record_dir = 'data/record'
# result
cfg.result_dir = 'data/result'
# trained model
cfg.trained_model_dir = 'data/trained_model'
cfg.trained_config_dir = 'data/trained_config'


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    cfg.exp_name = cfg.exp_name.replace('gittag', os.popen('git describe --tags --always').readline().strip())
    print('exp_name: ', cfg.exp_name, '\n')
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    cfg.trained_config_dir = os.path.join(cfg.trained_config_dir, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    cfg.local_rank = args.local_rank
    modules = [key for key in cfg if '_module' in key]
    for module in modules:
        cfg[module.replace('_module', '_path')] = cfg[module].replace('.', '/') + '.py'

def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--output_mesh', type=str, default='')
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
