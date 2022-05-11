import time
import datetime
import torch
import os
import open3d as o3d
from torch.nn.parallel import DistributedDataParallel
from lib.config import cfg
from lib.utils.data_utils import to_cuda
from lib.utils.mesh_utils import extract_mesh, refuse, transform


class Trainer(object):
    def __init__(self, network):
        print('GPU ID: ', cfg.local_rank)
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
            network = DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                # find_unused_parameters=True
           )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                #batch[k] = [b.cuda() for b in batch[k]]
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            else:
                # batch[k] = batch[k].cuda()
                batch[k] = batch[k].to(self.device)
        return batch
    
    def get_loss_weights(self, epoch):
        loss_weights = dict()

        loss_weights['rgb'] = cfg.loss.rgb_weight

        loss_weights['depth'] = cfg.loss.depth_weight
        for decay_epoch in cfg.loss.depth_weight_decay_epochs:
            if epoch >= decay_epoch:
                loss_weights['depth'] *= cfg.loss.depth_weight_decay
        if epoch >= cfg.loss.depth_loss_clamp_epoch:
            loss_weights['depth_loss_clamp'] = cfg.loss.depth_loss_clamp
        
        loss_weights['joint_start'] = epoch >= cfg.loss.joint_start
        loss_weights['joint'] = cfg.loss.joint_weight

        loss_weights['ce_cls'] = torch.tensor([cfg.loss.non_plane_weight, 1.0, 1.0])
        loss_weights['ce_cls'] = to_cuda(loss_weights['ce_cls'])

        loss_weights['ce'] = cfg.loss.ce_weight
        for decay_epoch in cfg.loss.ce_weight_decay_epochs:
            if epoch >= decay_epoch:
                loss_weights['ce'] *= cfg.loss.ce_weight_decay
        
        loss_weights['eikonal'] = cfg.loss.eikonal_weight

        return loss_weights

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        loss_weights = self.get_loss_weights(epoch)

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = to_cuda(batch, self.device)
            batch['loss_weights'] = loss_weights
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, save_mesh=True, evaluate_mesh=False, data_loader=None, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        mesh = extract_mesh(self.network.net.model.sdf_net)
        if save_mesh and not evaluate_mesh:
            os.makedirs(f'{cfg.result_dir}/', exist_ok=True)
            mesh.export(f'{cfg.result_dir}/{epoch}.obj')
        if evaluate_mesh:
            assert data_loader is not None
            assert evaluator is not None
            mesh = refuse(mesh, data_loader)
            mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)
            mesh_gt = o3d.io.read_triangle_mesh(f'{cfg.test_dataset.data_root}/{cfg.test_dataset.scene}/gt.obj')
            evaluate_result = evaluator.evaluate(mesh, mesh_gt)
            print(evaluate_result)
