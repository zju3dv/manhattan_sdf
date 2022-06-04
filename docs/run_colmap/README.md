# Run COLMAP

## Requirements

Compile [this customized version of COLMAP](https://github.com/B1ueber2y/colmap/tree/c84269d693246d8294307cc32f851813f18b6a2d), which is a submodule of [NerfingMVS](https://github.com/weiyithu/NerfingMVS).

## Run

```shell
python run.py
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{schoenberger2016sfm,
    author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
    title={Structure-from-Motion Revisited},
    booktitle={CVPR},
    year={2016},
}

@inproceedings{wei2021nerfingmvs,
  author={Wei, Yi and Liu, Shaohui and Rao, Yongming and Zhao, Wang and Lu, Jiwen and Zhou, Jie},
  title={NerfingMVS: Guided Optimization of Neural Radiance Fields for Indoor Multi-view Stereo},
  booktitle ={ICCV},
  year={2021}
}

@inproceedings{guo2022manhattan,
  title={Neural 3D Scene Reconstruction with the Manhattan-world Assumption},
  author={Guo, Haoyu and Peng, Sida and Lin, Haotong and Wang, Qianqian and Zhang, Guofeng and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2022}
}
```