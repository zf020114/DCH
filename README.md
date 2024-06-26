## Abstract

高分辨率遥感图像的可获得性大幅提升，使得遥感图像目标精细化检测成为了遥感以及计算机视觉领域重要的研究方向。针对遥感图像目标精细化检测中存在的相似数据利用不充分、错误标签影响模型精度和相似类别难以区分的问题，本文提出了一种基于双分类头的遥感图像精细化目标检测方法。首先，针对遥感图像精细化目标检测中无法有效利用相似数据的问题，提出了一种双分类检测头，不同的分类头分别对不同数据集训练，让类别定义不同的相似数据共同参与训练，进而有效利用相似数据，显著提升了模型精度。其次，针对训练标签噪声问题，设计了一种基于预测的错误标签过滤方法，减小错误标签对模型训练的影响。最后，针对精细化目标检测中类内差异大、类间差异小的问题，定义了一种Margin交叉熵损失，通过增大分类边界提高了模型精度。在精细化遥感目标检测竞赛数据集和FAIR1M数据集上的实验表明，本文提出的方法显著提高了遥感影像目标精细化检测的精度和鲁棒性。

## Introduction

This repository is the official implementation of "A Fine-grained Obect Detection Method for Remote Sensing Images Based on Dual Classification Head" 
The master branch is built on MMRotate which works with **PyTorch 1.6+**.

LSKNet backbone code is placed under mmrotate/models/backbones/, and the train/test configure files are placed under configs/lsknet/ 


## Results and models


精细化目标检测竞赛数据集

|                           Model                            |  mAP  | FPS | lr schd | Batch Size |                                   Configs                                    |                                                               Download                                                               |     note     |
| :--------------------------------------------------------: | :---: | :---: | :-----: | :--------: | :--------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :----------: |
|                  双分类头+错误标签过滤+ Margin交叉熵损失                   | 75.7 | 36.2  |   1x    |    10    |     [sk_double_filter_t_fpn_1x_yami_le90_fp16_r75_classblance6](./configs/lsknet
/lsk_double_filter_t_fpn_1x_yami_le90_fp16_r75_classblance6.py)     | [model] | [log]|              |



## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/zcablii/Large-Selective-Kernel-Network.git
cd Large-Selective-Kernel-Network
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)





## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation


## License

This project is released under the [Apache 2.0 license](LICENSE).
