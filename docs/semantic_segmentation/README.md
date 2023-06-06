# Semantic Segmentation

To implement 2D semantic segmentation for indoor scenes, we train [DeepLabV3+](https://arxiv.org/abs/1802.02611) based on [Detectron2](https://github.com/facebookresearch/detectron2). We use the [training set of ScanNetV2](https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/scannetv2_train.txt) for training and we use [NYU label IDs](http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt).

## Requirements

You need to install Detectron2, and we recommend the 0.4 version with PyTorch 1.6, please refer to the [installation instructions](https://detectron2.readthedocs.io/en/v0.4.1/tutorials/install.html).

## Run

We provide the trained model, you can download it from [here](https://drive.google.com/file/d/1sbboJHCMmFb1xzFBsf0ynfixJ1YcvT-r/view?usp=sharing) and run inference by:
```shell
python inference.py
```


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{chen2018encoder,
  title={Encoder-decoder with atrous separable convolution for semantic image segmentation},
  author={Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  booktitle={ECCV},
  year={2018}
}

@misc{wu2019detectron2,
  author={Yuxin Wu and Alexander Kirillov and Francisco Massa and Wan-Yen Lo and Ross Girshick},
  title={Detectron2},
  howpublished={\url{https://github.com/facebookresearch/detectron2}},
  year={2019}
}

@inproceedings{guo2022manhattan,
  title={Neural 3D Scene Reconstruction with the Manhattan-world Assumption},
  author={Guo, Haoyu and Peng, Sida and Lin, Haotong and Wang, Qianqian and Zhang, Guofeng and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2022}
}
```