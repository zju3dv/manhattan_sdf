import numpy as np
import torch
import torch.nn as nn
from torch import autograd

from lib.config import cfg


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class SDFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = cfg.model.sdf.net_depth
        self.W = cfg.model.net_width
        self.W_geo_feat = cfg.model.feature_width
        self.skips = cfg.model.sdf.skips
        embed_multires = cfg.model.sdf.fr_pos
        self.embed_fn, input_ch = get_embedder(embed_multires)

        surface_fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.D+1):
            # decide out_dim
            if l == self.D:
                if self.W_geo_feat > 0:
                    out_dim = 1 + self.W_geo_feat
                else:
                    out_dim = 1
            elif (l+1) in self.skips:
                out_dim = self.W - input_ch  # recude output dim before the skips layers, as in IDR / NeuS
            else:
                out_dim = self.W
                
            # decide in_dim
            if l == 0:
                in_dim = input_ch
            else:
                in_dim = self.W
            
            if l != self.D:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus(beta=100))
            else:
                layer = nn.Linear(in_dim, out_dim)

            # if true preform preform geometric initialization
            if cfg.model.sdf.geometric_init:
                #--------------
                # sphere init, as in SAL / IDR.
                #--------------
                if l == self.D:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias, -cfg.model.sdf.radius_init) 
                elif embed_multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)   # let the initial weights for octaves to be 0.
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif embed_multires > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):], 0.0) # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if cfg.model.sdf.weight_norm:
                layer = nn.utils.weight_norm(layer)

            surface_fc_layers.append(layer)

        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)

    def forward(self, x: torch.Tensor, return_h = False):
        x = self.embed_fn(x)
        
        h = x
        for i in range(self.D):
            if i in self.skips:
                # NOTE: concat order can not change! there are special operations taken in intialization.
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.surface_fc_layers[i](h)
        
        out = self.surface_fc_layers[-1](h)
        
        if self.W_geo_feat > 0:
            h = out[..., 1:]
            out = out[..., :1].squeeze(-1)
        else:
            out = out.squeeze(-1)
        
        out = -out  # make it suitable to inside-out scene

        if return_h:
            return out, h
        else:
            return out
    
    def forward_with_nablas(self,  x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sdf, h = self.forward(x, return_h=True)
            nabla = autograd.grad(
                sdf,
                x,
                torch.ones_like(sdf, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        if not has_grad:
            sdf = sdf.detach()
            nabla = nabla.detach()
            h = h.detach()
        return sdf, nabla, h


class RadianceNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        input_ch_pts = 3
        input_ch_views = 3
        self.skips = cfg.model.radiance.skips
        self.D = cfg.model.radiance.net_depth
        self.W = cfg.model.net_width
        embed_multires = cfg.model.radiance.fr_pos
        embed_multires_view = cfg.model.radiance.fr_view
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view)
        self.W_geo_feat = cfg.model.feature_width
        in_dim_0 = input_ch_pts + input_ch_views + 3 + self.W_geo_feat
        
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.D + 1):
            # decicde out_dim
            if l == self.D:
                out_dim = 3
            else:
                out_dim = self.W
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + self.W
            else:
                in_dim = self.W
            
            if l != self.D:
                layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
            
            if cfg.model.radiance.weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        view_dirs: torch.Tensor, 
        normals: torch.Tensor, 
        geometry_feature: torch.Tensor
    ):
        # calculate radiance field
        x = self.embed_fn(x)
        view_dirs = self.embed_fn_view(view_dirs)
        radiance_input = torch.cat([x, view_dirs, normals, geometry_feature], dim=-1)
        
        h = radiance_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h


class SemanticNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        input_ch_pts = 3
        self.skips = cfg.model.semantic.skips
        self.D = cfg.model.semantic.net_depth
        self.W = cfg.model.net_width
        embed_multires = cfg.model.semantic.fr_pos
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        self.W_geo_feat = cfg.model.feature_width
        in_dim_0 = input_ch_pts + self.W_geo_feat
        
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(self.D + 1):
            # decicde out_dim
            if l == self.D:
                out_dim = 3
            else:
                out_dim = self.W
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + self.W
            else:
                in_dim = self.W
            
            if l != self.D:
                layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
            
            if cfg.model.semantic.weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        geometry_feature: torch.Tensor):
        # calculate semantic field
        x = self.embed_fn(x)
        semantic_input = torch.cat([x, geometry_feature], dim=-1)
        
        h = semantic_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, semantic_input], dim=-1)
            h = self.layers[i](h)
        return h
