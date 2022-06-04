import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from detectron2.engine import default_argument_parser
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

height, width = 480, 640


def load_img(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height)) / 255
    return img


def predict(input_img, predictor):
    sem_seg = predictor(input_img)['sem_seg']
    sem_seg = F.softmax(sem_seg, dim=0)
    for i in range(41):
        if i in [0, 1, 2]:
            pass
        elif i == 8: # regard door as wall
            sem_seg[1] += sem_seg[i]
        elif i == 30: # regard white board as wall
            sem_seg[1] += sem_seg[i]
        elif i == 20: # regard floor mat as floor
            sem_seg[2] += sem_seg[i]
        else:
            sem_seg[0] += sem_seg[i]
    sem_seg = sem_seg[[0, 1, 2]]
    score, sem_seg = sem_seg.max(dim=0)
    sem_seg = sem_seg.cpu().numpy()
    return sem_seg


def main(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file('./configs/deeplabv3plus.yaml')
    cfg.MODEL.WEIGHTS = './model.pth'
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 41
    cfg.MODEL.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.MODEL.PIXEL_STD = [0.5, 0.5, 0.5]
    predictor = DefaultPredictor(cfg)

    for data_root in [
        '../../data/scannet/0050_00/'  # TODO: modify this to your path
    ]:
        img_path = f'{data_root}/images/'
        semantic_path = f'{data_root}/semantic_deeplab/'
        os.makedirs(semantic_path, exist_ok=True)
        imgs = os.listdir(img_path)
        for img_filename in tqdm(imgs):
            img = load_img(f'{img_path}/{img_filename}')
            sem_seg = predict(img, predictor)
            sem_seg = (sem_seg * 80).astype(np.uint8)
            sem_seg = cv2.resize(sem_seg, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f'{semantic_path}/{img_filename[:-4]}.png', sem_seg)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
