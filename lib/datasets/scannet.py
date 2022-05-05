import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
from lib.config import cfg


WALL_SEMANTIC_ID = 80
FLOOR_SEMANTIC_ID = 160


class Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], kwargs['scene']
        self.instance_dir = f'{data_root}/{scene}'
        self.split = split
        assert os.path.exists(self.instance_dir)

        image_dir = '{0}/images'.format(self.instance_dir)
        self.image_list = os.listdir(image_dir)
        self.image_list.sort(key=lambda _:int(_.split('.')[0]))
        self.n_images = len(self.image_list)
        
        self.intrinsic_all = []
        self.c2w_all = []
        self.rgb_images = []

        self.semantic_deeplab = []
        self.depth_colmap = []

        intrinsic = np.loadtxt(f'{self.instance_dir}/intrinsic.txt')[:3, :3]

        for imgname in tqdm(self.image_list, desc='Loading dataset'):
            c2w = np.loadtxt(f'{self.instance_dir}/pose/{imgname[:-4]}.txt')
            self.c2w_all.append(c2w)
            self.intrinsic_all.append(intrinsic)

            rgb = cv2.imread(f'{self.instance_dir}/images/{imgname[:-4]}.png')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = (rgb.astype(np.float32) / 255).transpose(2, 0, 1)
            _, self.H, self.W = rgb.shape
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(rgb)

            if self.split == 'train':
                depth_path = f'{self.instance_dir}/depth_colmap/{imgname[:-4]}.npy'
                if os.path.exists(depth_path):
                    depth_colmap = np.load(depth_path)
                    depth_colmap[depth_colmap > 2.0] = 0
                else:
                    depth_colmap = np.zeros((self.H, self.W), np.float32)
                depth_colmap = depth_colmap.reshape(-1)
                self.depth_colmap.append(depth_colmap)
                
                semantic_deeplab = cv2.imread(f'{self.instance_dir}/semantic_deeplab/{imgname[:-4]}.png', -1)
                semantic_deeplab = semantic_deeplab.reshape(-1)
                wall_mask = semantic_deeplab == WALL_SEMANTIC_ID
                floor_mask = semantic_deeplab == FLOOR_SEMANTIC_ID
                bg_mask = ~(wall_mask | floor_mask)
                semantic_deeplab[wall_mask] = 1
                semantic_deeplab[floor_mask] = 2
                semantic_deeplab[bg_mask] = 0
                self.semantic_deeplab.append(semantic_deeplab)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        c2w, intrinsic = self.c2w_all[idx], self.intrinsic_all[idx]

        ret = {'rgb': self.rgb_images[idx]}

        if self.split == 'train':
            rays = self.gen_rays(c2w, intrinsic)
            ret['rays'] = rays
            ret['semantic_deeplab'] = self.semantic_deeplab[idx]
            ret['depth_colmap'] = self.depth_colmap[idx]

            ids = np.random.choice(len(rays), cfg.train.N_rays, replace=False)
            for k in ret:
                ret[k] = ret[k][ids]
        
        else:
            ret['c2w'] = c2w
            ret['intrinsic'] = intrinsic

        ret.update({'meta': {'h': self.H, 'w': self.W, 'filename': self.image_list[idx]}})
        return ret

    def gen_rays(self, c2w, intrinsic):
        H, W = self.H, self.W
        rays_o = c2w[:3, 3]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
        XYZ = XYZ @ np.linalg.inv(intrinsic).T
        XYZ = XYZ @ c2w[:3, :3].T
        rays_d = XYZ.reshape(-1, 3)

        rays = np.concatenate([rays_o[None].repeat(H*W, axis=0), rays_d], axis=-1)
        return rays.astype(np.float32)
