from .base import SDFNet, RadianceNet, SemanticNet
from .ray_sampler import sdf_to_sigma, fine_sample

import copy
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.net_utils import batchify_query
from lib.config import cfg


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.speed_factor = cfg.model.speed_factor
        ln_beta_init = np.log(cfg.model.beta_init) / self.speed_factor
        self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)

        self.sdf_net = SDFNet()
        self.radiance_net = RadianceNet()
        self.semantic_net = SemanticNet()

    def forward_ab(self):
        beta = torch.exp(self.ln_beta * self.speed_factor)
        return 1./beta, beta

    def forward_surface(self, x: torch.Tensor):
        sdf = self.sdf_net.forward(x)
        return sdf        

    def forward_surface_with_nablas(self, x: torch.Tensor):
        sdf, nablas, h = self.sdf_net.forward_with_nablas(x)
        return sdf, nablas, h

    def forward(self, x:torch. Tensor, view_dirs: torch.Tensor):
        sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        semantics = self.semantic_net.forward(x, geometry_feature)
        return radiances, semantics, sdf, nablas
    
    def forward_semantic(self, x:torch. Tensor):
        sdf, nablas, geometry_feature = self.forward_surface_with_nablas(x)
        semantics = self.semantic_net.forward(x, geometry_feature)
        return semantics


def volume_render(
    rays_o, 
    rays_d,
    model: MLP,
    near=0.0,
    far=2.0,
    perturb = True,
    ):

    device = rays_o.device
    rayschunk = cfg.sample.rayschunk
    netchunk = cfg.sample.netchunk
    N_samples = cfg.sample.N_samples
    N_importance = cfg.sample.N_importance
    max_upsample_steps = cfg.sample.max_upsample_steps
    max_bisection_steps = cfg.sample.max_bisection_steps
    epsilon = cfg.sample.epsilon

    DIM_BATCHIFY = 1
    B = rays_d.shape[0]  # batch_size
    flat_vec_shape = [B, -1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()

    depth_ratio = rays_d.norm(dim=-1)
    rays_d = F.normalize(rays_d, dim=-1)
    
    batchify_query_ = functools.partial(batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)

    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):

        view_dirs = rays_d
        
        prefix_batch = [B]
        N_rays = rays_o.shape[-2]
        
        nears = near * torch.ones([*prefix_batch, N_rays, 1]).to(device)
        fars = far * torch.ones([*prefix_batch, N_rays, 1]).to(device)

        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = nears * (1 - _t) + fars * _t
        alpha, beta = model.forward_ab()
        with torch.no_grad():
            _t = torch.linspace(0, 1, N_samples*4).float().to(device)
            d_init = nears * (1 - _t) + fars * _t
            
            d_fine, beta_map, iter_usage = fine_sample(
                model.forward_surface, d_init, rays_o, rays_d, 
                alpha_net=alpha, beta_net=beta, far=fars, 
                eps=epsilon, max_iter=max_upsample_steps, max_bisection=max_bisection_steps, 
                final_N_importance=N_importance, perturb=perturb, 
                N_up=N_samples*4
            )

        d_all = torch.cat([d_coarse, d_fine], dim=-1)
        d_all, _ = torch.sort(d_all, dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        
        radiances, semantics, sdf, nablas = batchify_query_(model.forward, pts, view_dirs.unsqueeze(-2).expand_as(pts))
        sigma = sdf_to_sigma(sdf, alpha, beta)
            
        delta_i = d_all[..., 1:] - d_all[..., :-1]
        p_i = torch.exp(-F.relu_(sigma[..., :-1] * delta_i))

        tau_i = (1 - p_i + 1e-10) * (
            torch.cumprod(
                torch.cat(
                    [torch.ones([*p_i.shape[:-1], 1], device=device), p_i], dim=-1), 
                dim=-1)[..., :-1]
            )

        rgb_map = torch.sum(tau_i[..., None] * radiances[..., :-1, :], dim=-2)
        semantic_map = torch.sum(tau_i[..., None] * semantics[..., :-1, :], dim=-2)
        
        distance_map = torch.sum(tau_i / (tau_i.sum(-1, keepdim=True)+1e-10) * d_all[..., :-1], dim=-1)
        depth_map = distance_map / depth_ratio
        acc_map = torch.sum(tau_i, -1)

        ret_i = OrderedDict([
            ('rgb', rgb_map),
            ('semantic', semantic_map),
            ('distance', distance_map),
            ('depth', depth_map),
            ('mask_volume', acc_map)
        ])

        surface_points = rays_o + rays_d * distance_map[..., None]
        _, surface_normals, _ = model.sdf_net.forward_with_nablas(surface_points.detach())
        ret_i['surface_normals'] = surface_normals

        # normals_map = F.normalize(nablas, dim=-1)
        # N_pts = min(tau_i.shape[-1], normals_map.shape[-2])
        # normals_map = (normals_map[..., :N_pts, :] * tau_i[..., :N_pts, None]).sum(dim=-2)
        # ret_i['normals_volume'] = normals_map

        ret_i['sdf'] = sdf
        ret_i['nablas'] = nablas
        ret_i['radiance'] = radiances
        ret_i['alpha'] = 1.0 - p_i
        ret_i['p_i'] = p_i
        ret_i['visibility_weights'] = tau_i
        ret_i['d_vals'] = d_all
        ret_i['sigma'] = sigma
        ret_i['beta_map'] = beta_map
        ret_i['iter_usage'] = iter_usage

        return ret_i
        
    ret = {}
    for i in range(0, rays_o.shape[DIM_BATCHIFY], rayschunk):
        ret_i = render_rayschunk(rays_o[:, i:i+rayschunk], rays_d[:, i:i+rayschunk])
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)
    
    alpha, beta = model.forward_ab()
    alpha, beta = alpha.data, beta.data
    ret['scalars'] = {'alpha': alpha, 'beta': beta}

    return ret


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = MLP()
        
        self.theta = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        # <cos(theta), sin(tehta), 0> is $\mathbf{n}_w$ in equation (9)
    
    def forward(self, batch):
        rays = batch['rays']
        rays_o, rays_d = rays[:, :, :3], rays[:, :, 3:6]
        rays_d[rays_d.abs() < 1e-6] = 1e-6

        if self.training:
            near = cfg.train_dataset.near
            far = cfg.train_dataset.far
            pertube = True
        else:
            near = cfg.test_dataset.near
            far = cfg.test_dataset.far
            pertube = False

        return volume_render(
            rays_o,
            rays_d,
            self.model,
            near = near,
            far=far,
            perturb=pertube
        )
