# CoAlign (ICRA2023)

Robust Collaborative 3D Object Detection in Presence of Pose Errors 

[Paper](https://arxiv.org/abs/2211.07214)

## New features (Compared with OpenCOOD):

- Dataset Support
  - [x] OPV2V
  - [x] V2X-Sim 2.0
  - [x] DAIR-V2X
  - [x] V2XSet

- SOTA collaborative perception method support
    - [x] [Attentive Fusion [ICRA2022]](https://arxiv.org/abs/2109.07644)
    - [x] [Cooper [ICDCS]](https://arxiv.org/abs/1905.05265)
    - [x] [F-Cooper [SEC2019]](https://arxiv.org/abs/1909.06459)
    - [x] [V2VNet [ECCV2022]](https://arxiv.org/abs/2008.07519)
    - [x] [FPV-RCNN [RAL2022]](https://arxiv.org/pdf/2109.11615.pdf)
    - [x] [DiscoNet [NeurIPS2021]](https://arxiv.org/abs/2111.00643)
    - [x] [V2X-ViT [ECCV2022]](https://github.com/DerrickXuNu/v2x-vit) 
    - [x] [MASH [IROS 2021]](https://arxiv.org/abs/2107.00771)
    - [x] [V2VNet(robust) [CoRL2020]](https://arxiv.org/abs/2011.05289)
    - [x] [CoAlign [ICRA2023]](https://arxiv.org/abs/2211.07214)

- Visualization support
  - [x] BEV visualization
  - [x] 3D visualization

- 1-round/2-round communication support
  - transform point cloud first (2-round communication)
  - warp feature map (1-round communication, by default in this repo.)

- Pose error simulation support

## Installation

I recommend you visit [CoAlign Installation Guide](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie) to learn how to install this repo. I would supplement the English version and the usage of CoAlign into readme soon.

Or you can refer to [OpenCOOD data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [OpenCOOD installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare
data and install CoAlign. The installation is totally the same as OpenCOOD.

The api of camera collaboration and the implementation Lift-Splat will be released in March. OPV2V, V2XSet and DAIR-V2X are supported. Pay attention to the repo if you need.

> Note that I update the AP calculation (sorted by confidence) and add data augmentations (reinitialize) in this codebase, so the result will be higher than that reported in the current paper. I retrain all the models and would update the paper to the final version before March. Then I will remove this paragraph.




## Complemented Annotations for DAIR-V2X-C
Originally DAIR-V2X only annotates 3D boxes within the range of camera's view in vehicle-side. We supplement the missing 3D box annotations to enable the 360 degree detection. With fully complemented vehicle-side labels, we regenerate the cooperative labels for users, which follow the original cooperative label format.

Original Annotations | Complemented Annotations 
---|---
![Original1](images/dair-v2x_compare_gif/before1.gif) | ![Complemented1](images/dair-v2x_compare_gif/after1.gif)
![Original2](images/dair-v2x_compare_gif/before2.gif) | ![Complemented2](images/dair-v2x_compare_gif/after2.gif)
![Original3](images/dair-v2x_compare_gif/before3.gif) | ![Complemented3](images/dair-v2x_compare_gif/after3.gif)


**Download:** [Google Drive](https://drive.google.com/file/d/13g3APNeHBVjPcF-nTuUoNOSGyTzdfnUK/view?usp=sharing)

**Website:** [Website](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/)

## Citation
```
@article{lu2022robust,
  title={Robust Collaborative 3D Object Detection in Presence of Pose Errors},
  author={Lu, Yifan and Li, Quanhao and Liu, Baoan and Dianati, Mehrdad and Feng, Chen and Chen, Siheng and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2211.07214},
  year={2022}
}
```

## Acknowlege

This project is impossible without the code of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), [g2opy](https://github.com/uoip/g2opy) and [d3d](https://github.com/cmpute/d3d)!

Thanks again to [@DerrickXuNu](https://github.com/DerrickXuNu)
 for the great code framework.