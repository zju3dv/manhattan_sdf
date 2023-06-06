import os
import cv2
from tqdm import tqdm
import numpy as np
import struct

# from https://github.com/colmap/colmap/tree/dev/scripts/python
from utils.read_write_model import read_images_binary, rotmat2qvec
from utils.read_write_dense import read_array
from utils.database import COLMAPDatabase


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
scale_radius = 1.0
height=480
width=640


def load_point_vis(path, masks):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        print('point number: {}'.format(n))
        for i in range(n):
            m = struct.unpack('<I', f.read(4))[0]
            for j in range(m):
                idx, u, v = struct.unpack('<III', f.read(4 * 3))
                masks[idx][v, u] = 1


for scene_id in ['0050_00']:
    source = f'/home/guohaoyu/data/scannet/scene{scene_id}/'  # TODO: modify this to your path
    target = f'../../data/scannet/{scene_id}'

    os.makedirs(f'{target}/images', exist_ok=True)
    os.makedirs(f'{target}/pose', exist_ok=True)
    os.makedirs(f'{target}/depth_colmap', exist_ok=True)

    id_list = os.listdir(source + 'color')
    id_list = [id[:-4] for id in id_list if id.endswith('0.jpg')]
    id.sort(key=lambda _:int(_[:-4]))

    pose_dict = dict()
    for id in id_list:
        pose_dict[id] = np.loadtxt(source + f'pose/{id}.txt')

    id_list = [id for id in id_list if not np.isinf(pose_dict[id]).any()]
    id_list.sort()

    translation_list = []
    for id in id_list:
        translation_list.append(pose_dict[id][None, :3, 3])
    translation_list = np.concatenate(translation_list)
    translation_center = (translation_list.max(axis=0) + translation_list.min(axis=0)) / 2
    translation_list -= translation_center
    max_cam_norm = np.linalg.norm(translation_list, axis=1).max()
    scale = (scale_radius / max_cam_norm / 1.1)

    for id in id_list:
        pose_dict[id][:3, 3] -= translation_center
        pose_dict[id][:3, 3] *= scale

    with open(f'{target}/offset.txt', 'w') as f:
        f.write(f'{translation_center}')

    with open(f'{target}/scale.txt', 'w') as f:
        f.write(f'{scale}')

    os.system(f'cp {source}/intrinsic/intrinsic_depth.txt {target}/intrinsic.txt')

    for id in tqdm(id_list):
        color = cv2.imread(f'{source}/color/{id}.jpg')
        color = cv2.resize(color, (width, height))
        cv2.imwrite(f'{target}/images/{id}.png', color)
        np.savetxt(f'{target}/pose/{id}.txt', pose_dict[id])

    colmap_bin = '/home/guohaoyu/repos/NerfingMVS/colmap/build/src/exe/colmap'  # TODO: modify this to your path

    data_list = []
    for i, id in enumerate(id_list):
        rt = pose_dict[id]
        rt = np.linalg.inv(rt)
        r = rt[:3, :3]
        t = rt[:3, 3]
        q = rotmat2qvec(r)
        data = [i + 1, *q, *t, 1, f'{id}.png']
        data = [str(_) for _ in data]
        data = ' '.join(data)
        data_list.append(data)

    os.makedirs(f'{target}/model/', exist_ok=True)
    os.system(f'touch {target}/model/points3D.txt')

    intrinsic = np.loadtxt(f'{target}/intrinsic.txt')

    with open(f'{target}/model/cameras.txt', 'w') as f:
        f.write(f'1 PINHOLE {width} {height} {intrinsic[0][0]} {intrinsic[1][1]} {intrinsic[0][2]} {intrinsic[1][2]}')

    with open(f'{target}/model/images.txt', 'w') as f:
        for data in data_list:
            f.write(data)
            f.write('\n\n')

    os.system(f'{colmap_bin} feature_extractor --database_path {target}/database.db --image_path {target}/images > {target}/colmap_output.txt')

    os.system(f'{colmap_bin} exhaustive_matcher --database_path {target}/database.db > {target}/colmap_output.txt')

    db = COLMAPDatabase.connect(f'{target}/database.db')

    images = list(db.execute('select * from images'))

    data_list = []
    for image in images:
        id = image[1][:-4]
        rt = pose_dict[id]
        rt = np.linalg.inv(rt)
        r = rt[:3, :3]
        t = rt[:3, 3]
        q = rotmat2qvec(r)
        data = [image[0], *q, *t, 1, f'{id}.png']
        data = [str(_) for _ in data]
        data = ' '.join(data)
        data_list.append(data)

    with open(f'{target}/model/images.txt', 'w') as f:
        for data in data_list:
            f.write(data)
            f.write('\n\n')
    
    os.makedirs(f'{target}/sparse/', exist_ok=True)

    os.system(f'{colmap_bin} point_triangulator --database_path {target}/database.db --image_path {target}/images --input_path {target}/model --output_path {target}/sparse > {target}/colmap_output.txt')

    os.system(f'{colmap_bin} image_undistorter --image_path {target}/images --input_path {target}/sparse --output_path {target}/dense > {target}/colmap_output.txt')

    os.system(f'{colmap_bin} patch_match_stereo --workspace_path {target}/dense --PatchMatchStereo.cache_size 64 > {target}/colmap_output.txt')

    os.system(f'{colmap_bin} stereo_fusion --workspace_path {target}/dense --output_path {target}/dense/fuse.ply --StereoFusion.cache_size 64 > {target}/colmap_output.txt')

    images_bin_path = f'{target}/sparse/images.bin'
    images = read_images_binary(images_bin_path)
    names = [_[1].name for _ in images.items()]

    shape = (height, width)

    ply_vis_path = f'{target}/dense/fuse.ply.vis'
    assert os.path.exists(ply_vis_path)
    masks = [np.zeros(shape, dtype=np.uint8) for name in names]
    load_point_vis(ply_vis_path, masks)
    
    for name, mask in tqdm(zip(names, masks)):
        depth_bin_path = f'{target}/dense/stereo/depth_maps/{name}.geometric.bin'
        if not os.path.exists(depth_bin_path):
            continue
        depth_fname = depth_bin_path
        depth = read_array(depth_fname)
        # depth[mask == 0] = 0
        np.save(f'{target}/depth_colmap/{name[:-4]}.npy', depth)
