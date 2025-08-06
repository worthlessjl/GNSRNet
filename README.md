
# GNSRNet: A Geometric Guided Noise Reduction Super-Resolution Network for Remote Sensing Tiny Object Detection


<p align='center'>
  <img src='GNSR.jpg' width="800px">
</p>



## Abstract

Tiny objects in remote sensing typically face the challenges of being submerged in complex backgrounds, limited feature representation, and high sensitivity to prediction errors due to their small size and diverse shape. These challenges make tiny object detection a significant difficulty, and traditional object detection methods often yield poor performance. To tackle irrelevant information interference, insufficient feature representation, and neglect of the objects' geometric characteristics, a Geometric Guided Noise Reduction Super-Resolution Network is proposed. First, an adaptive dynamic noise reduction module is introduced to fundamentally mitigate spatial misalignment in feature fusion by effectively suppressing the noise arising from the upsampling process. Second, a coupled-training and decoupled-detection Dual-Stream Progressive Super-Resolution detection head is incorporated. The head reduces the receptive field to precisely align with tiny object dimensions and employs a weight-sharing mechanism to implicitly learn super-resolution features. Furthermore, a novel progressive loss annealing strategy is utilized to reduce the dependency of the super-resolution branch. Third, a geometric characteristic regression metric is proposed. This metric comprehensively considers the location prediction and the shape similarity between the prediction and ground truth boxes. By enhancing the quality of prediction boxes, which improve the accuracy of prediction. Extensive experiments were produced on the remote sensing tiny object dataset AI-TOD v1, AI-TOD v2, USOD, and VisDrone, compared with other state-of-the-art (SOTA) methods, GNSRNet demonstrates superior performance on these benchmarks. Specifically, it reaches an AP of 31.4 on AI-TOD v1, 30.4 on AI-TOD v2, 37.4 on USOD, and 30.3 on the VisDrone, which achieves SOTA performance.


## Requirements

This project maintains the same environment requirements as YOLOv11. You need install the `ultralytics` package, including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml). 
```bash
pip install ultralytics
```
Pytorch>=1.8

Python>=3.8

## Dataset

The dataset directory should follow the standard YOLO format:

```bash
datasets
├── USOD
│   ├── images
│   │   ├── train
│   │   │   ├── img001.jpg
│   │   │   └── ...
│   │   ├── val
│   │   │   ├── img101.jpg
│   │   │   └── ...
│   └── labels
│       ├── train
│       │   ├── img001.txt
│       │   └── ...
│       ├── val
│       │   ├── img101.txt
│       │   └── ...
```

## Usage

1.You can directly use `train.py` to train your model.

2.The `val.py` script is used to validate the model performance.

3.The pretrained weights for the datasets are contained in the `runs` directory.

