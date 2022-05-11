import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net

    def forward(self, batch):
        output = self.net(batch)
        if not self.net.training:
            return output

        loss_weights = batch['loss_weights']
        loss = 0
        scalar_stats = {}

        rgb_loss = F.l1_loss(batch['rgb'], output['rgb'], reduction='none').mean() # Eq.5
        scalar_stats.update({'rgb_loss': rgb_loss})
        loss += loss_weights['rgb'] * rgb_loss

        depth_colmap_mask = batch['depth_colmap'] > 0
        if depth_colmap_mask.sum() > 0:
            depth_loss = F.l1_loss(output['depth'][depth_colmap_mask], batch['depth_colmap'][depth_colmap_mask], reduction='none') # Eq.7
            if 'depth_loss_clamp' in loss_weights:
                depth_loss = depth_loss.clamp(max=loss_weights['depth_loss_clamp'])
            depth_loss = depth_loss.mean()
            scalar_stats.update({'depth_loss': depth_loss})
            loss += loss_weights['depth'] * depth_loss

        semantic_deeplab = batch['semantic_deeplab']
        wall_mask = semantic_deeplab == 1
        floor_mask = semantic_deeplab == 2
        semantic_score_log = F.log_softmax(output['semantic'], dim=-1)
        semantic_score = torch.exp(semantic_score_log)

        surface_normals = output['surface_normals']
        surface_normals_normalized = F.normalize(surface_normals, dim=-1).clamp(-1., 1.)

        if loss_weights['joint_start']:
            bg_score, wall_score, floor_score = semantic_score.split(dim=-1, split_size=1)
            joint_loss = 0.

            if floor_mask.sum() > 0:
                floor_normals = surface_normals_normalized[floor_mask]
                floor_loss = (1 - floor_normals[..., 2]) # Eq.8
                joint_floor_loss = (floor_score[floor_mask][..., 0] * floor_loss).mean() # Eq.13
                joint_loss += joint_floor_loss
            
            if wall_mask.sum() > 0:
                wall_normals = surface_normals_normalized[wall_mask]
                wall_loss_vertical = wall_normals[..., 2].abs()
                theta = self.net.theta
                cos = wall_normals[..., 0] * torch.cos(theta) + wall_normals[..., 1] * torch.sin(theta)
                wall_loss_horizontal = torch.min(cos.abs(), torch.min((1 - cos).abs(), (1 + cos).abs())) # Eq.9
                wall_loss = wall_loss_vertical + wall_loss_horizontal
                joint_wall_loss = (wall_score[wall_mask][..., 0] * wall_loss).mean() # Eq.13
                joint_loss += joint_wall_loss
            
            if floor_mask.sum() > 0 or wall_mask.sum() > 0:
                scalar_stats.update({'joint_loss': joint_loss})
                loss += loss_weights['joint'] * joint_loss
            
        else: # Semantic score is unreliable in early training stage
            geo_loss = 0.

            if floor_mask.sum() > 0:
                floor_normals = surface_normals_normalized[floor_mask]
                floor_loss = (1 - floor_normals[..., 2]).mean()
                geo_loss += floor_loss
            
            if wall_mask.sum() > 0:
                wall_normals = surface_normals_normalized[wall_mask]
                wall_loss_vertical = wall_normals[..., 2].abs().mean()
                geo_loss += wall_loss_vertical

            if floor_mask.sum() > 0 or wall_mask.sum() > 0:
                scalar_stats.update({'geo_loss': geo_loss})
                loss += loss_weights['joint'] * geo_loss

        cross_entropy_loss = F.nll_loss(
            semantic_score_log.reshape(-1, 3),
            semantic_deeplab.reshape(-1).long(),
            weight=loss_weights['ce_cls']
        ) # Eq.14
        scalar_stats.update({'cross_entropy_loss': cross_entropy_loss})
        loss += loss_weights['ce'] * cross_entropy_loss

        nablas: torch.Tensor = output['nablas']
        _, _ind = output['visibility_weights'][..., :nablas.shape[-2]].max(dim=-1)
        nablas = torch.gather(nablas, dim=-2, index=_ind[..., None, None].repeat([*(len(nablas.shape)-1)*[1], 3]))
        eik_bounding_box = cfg.model.bounding_radius
        eikonal_points = torch.empty_like(nablas).uniform_(-eik_bounding_box, eik_bounding_box).to(nablas.device)
        _, nablas_eik, _ = self.net.model.sdf_net.forward_with_nablas(eikonal_points)
        nablas = torch.cat([nablas, nablas_eik], dim=-2)
        nablas_norm = torch.norm(nablas, dim=-1)
        eikonal_loss = F.mse_loss(nablas_norm, nablas_norm.new_ones(nablas_norm.shape), reduction='mean') # Eq.6
        scalar_stats.update({'eikonal_loss': eikonal_loss})
        loss += loss_weights['eikonal'] * eikonal_loss

        scalar_stats.update({'loss': loss})
        scalar_stats['beta'] = output['scalars']['beta']
        scalar_stats['theta'] = self.net.theta.data

        image_stats = {}

        return output, loss, scalar_stats, image_stats
