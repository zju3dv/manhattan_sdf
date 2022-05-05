import torch
import os
import torch.nn.functional
from termcolor import colored
from typing import Iterable
from lib.config import cfg


def load_model(net,
               optim,
               scheduler,
               recorder,
               model_dir,
               resume=True,
               epoch=-1):
               
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch
    print('Load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    net.load_state_dict(pretrained_model['net'])
    if 'optim' in pretrained_model:
        optim.load_state_dict(pretrained_model['optim'])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        recorder.load_state_dict(pretrained_model['recorder'])
        return pretrained_model['epoch'] + 1
    else:
        return 0


def save_model(net, optim, scheduler, recorder, model_dir, epoch, last=False):
    os.system('mkdir -p {}'.format(model_dir))
    model = {
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    if last:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) <= 5:
        return
    # os.system('rm {}'.format(
    #     os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('pretrained model does not exist', 'red'))
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth'
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0
        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('Load model: {}'.format(model_path))
    pretrained_model = torch.load(model_path)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    if 'epoch' in pretrained_model:
        return pretrained_model['epoch'] + 1
    else:
        return 0


def load_pretrain(net, model_dir):
    model_dir = os.path.join('data/trained_model', cfg.task, model_dir)
    if not os.path.exists(model_dir):
        return 1

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if pth != 'latest.pth']
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 1

    if 'latest.pth' in os.listdir(model_dir):
        pth = 'latest'
    else:
        pth = max(pths)

    print('Load pretrain model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    net.load_state_dict(pretrained_model['net'])
    return 0


def save_pretrain(net, task, model_dir):
    model_dir = os.path.join('data/trained_model', task, model_dir)
    os.system('mkdir -p ' +  model_dir)
    model = {'net': net.state_dict()}
    torch.save(model, os.path.join(model_dir, 'latest.pth'))


def batchify_query(query_fn, *args: Iterable[torch.Tensor], chunk, dim_batchify):
    # [(B), N_rays, N_pts, ...] -> [(B), N_rays*N_pts, ...]
    _N_rays = args[0].shape[dim_batchify]
    _N_pts = args[0].shape[dim_batchify+1]
    args = [arg.flatten(dim_batchify, dim_batchify+1) for arg in args]
    _N = args[0].shape[dim_batchify]
    raw_ret = []
    for i in range(0, _N, chunk):
        if dim_batchify == 0:
            args_i = [arg[i:i+chunk] for arg in args]
        elif dim_batchify == 1:
            args_i = [arg[:, i:i+chunk] for arg in args]
        elif dim_batchify == 2:
            args_i = [arg[:, :, i:i+chunk] for arg in args]
        else:
            raise NotImplementedError
        raw_ret_i = query_fn(*args_i)
        if not isinstance(raw_ret_i, tuple):
            raw_ret_i = [raw_ret_i]
        raw_ret.append(raw_ret_i)
    collate_raw_ret = []
    num_entry = 0
    for entry in zip(*raw_ret):
        if isinstance(entry[0], dict):
            tmp_dict = {}
            for list_item in entry:
                for k, v in list_item.items():
                    if k not in tmp_dict:
                        tmp_dict[k] = []
                    tmp_dict[k].append(v)
            for k in tmp_dict.keys():
                # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
                # tmp_dict[k] = torch.cat(tmp_dict[k], dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
                # NOTE: compatible with torch 1.6
                v = torch.cat(tmp_dict[k], dim=dim_batchify)
                tmp_dict[k] = v.reshape([*v.shape[:dim_batchify], _N_rays, _N_pts, *v.shape[dim_batchify+1:]])
            entry = tmp_dict
        else:
            # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
            # entry = torch.cat(entry, dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
            # NOTE: compatible with torch 1.6
            v = torch.cat(entry, dim=dim_batchify)
            entry = v.reshape([*v.shape[:dim_batchify], _N_rays, _N_pts, *v.shape[dim_batchify+1:]])
        collate_raw_ret.append(entry)
        num_entry += 1
    if num_entry == 1:
        return collate_raw_ret[0]
    else:
        return tuple(collate_raw_ret)
