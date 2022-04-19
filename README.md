# Monocular 3D Object Detection: An Extrinsic Parameter Free Approach

This repository is the official implementation of our paper.
For more details, please see our paper.

## Introduction
MonoEF is a **real-time** monocular 3D object detector for autonomous driving.  
Part of the code comes from [SMOKE](https://github.com/lzccccc/SMOKE),
[CenterNet](https://github.com/xingyizhou/CenterNet), 
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark),
and [Detectron2](https://github.com/facebookresearch/detectron2).


## Requirements
All codes are tested under the following environment:
*   Ubuntu 16.04
*   Python 3.7
*   Pytorch 1.3.1
*   CUDA 10.0

## Dataset
We train and test our model on official [KITTI 3D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Please first download the dataset and organize it as following structure:
```
kitti
│──training
│    ├──calib 
│    ├──label_2 
│    ├──image_2
│    └──ImageSets
└──testing
     ├──calib 
     ├──image_2
     └──ImageSets
```  

## Setup
1. We use `conda` to manage the environment:
```
conda create -n SMOKE python=3.7
```

2. Clone this repo:
```
git clone https://github.com/lzccccc/SMOKE
```

3. Build codes:
```
python setup.py build develop
```

4. Link to dataset directory:
```
mkdir datasets
ln -s /path_to_kitti_dataset datasets/kitti
```

## Getting started
First check the config file under `configs/`. 

We train the model on 4 GPUs with 32 batch size:
```
python tools/plain_train_net.py --num-gpus 4 --config-file "configs/smoke_gn_vector.yaml"
```

For single GPU training, simply run:
```
python tools/plain_train_net.py --config-file "configs/smoke_gn_vector.yaml"
```

We currently only support single GPU testing:
```
python tools/plain_train_net.py --eval-only --config-file "configs/smoke_gn_vector.yaml"
```

## Acknowledgement
[SMOKE](https://github.com/lzccccc/SMOKE)

[CenterNet](https://github.com/xingyizhou/CenterNet)

[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

[Detectron2](https://github.com/facebookresearch/detectron2)


## Citations
Please cite our paper if you find MonoEF is helpful for your research.
```
@inproceedings{zhou2021monocular,
  title={Monocular 3d object detection: An extrinsic parameter free approach},
  author={Zhou, Yunsong and He, Yuan and Zhu, Hongzi and Wang, Cheng and Li, Hongyang and Jiang, Qinhong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7556--7566},
  year={2021}
}
```
