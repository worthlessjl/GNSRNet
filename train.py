
from ultralytics import YOLO
model = YOLO("MYYOLOv2.yaml")
# 训练模型
train_results = model.train(
    data="/root/dataset/AI-TOD2/AITOD.yaml",  # 数据集 YAML 路径
    epochs=300, # 训练轮次
    imgsz=640,  # 训练图像尺寸
    batch=12,
    optimizer='SGD',
    lr0=0.01,
    lrf=0.2,
    momentum=0.9,
    warmup_epochs=3.0 ,
    warmup_momentum= 0.8,  # warmup initial momentum
    warmup_bias_lr=0.1,  # warmup initial bias lr
    box= 0.05,  # box loss gain
    cls= 0.5,# cls loss gain
    dfl=1.0,
    hsv_h= 0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s= 0.7,  # image HSV-Saturation augmentation (fraction)
    hsv_v= 0.4,  # image HSV-Value augmentation (fraction)
    degrees= 0.0 , # image rotation (+/- deg)
    translate= 0.1,  # image translation (+/- fraction)
    scale= 0.5 , # image scale (+/- gain)
    shear= 0.0 , # image shear (+/- deg)
    perspective= 0.0 , # image perspective (+/- fraction), range 0-0.001
    flipud= 0.0 , # image flip up-down (probability)
    fliplr= 0.5  ,# image flip left-right (probability)
    mosaic= 1.0,  # image mosaic (probability)
    mixup= 0.0 , # image mixup (probability)
    copy_paste= 0.0  ,# segment copy-paste (probability)
    close_mosaic= 10,
    device="0,1",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu

)
