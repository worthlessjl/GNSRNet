from ultralytics import YOLO
model = YOLO("/root/multiyolo/ultralytics-main/runs/detect/AITOD/weights/best.pt")  # load your custom model
# Validate the model
metrics = model.val(data="/root/dataset/AI-TOD2/AITOD.yaml"
 )

