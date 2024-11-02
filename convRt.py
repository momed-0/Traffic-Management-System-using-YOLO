from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")


success = model.export(format="engine",device=0,half=True,imgsz=640)  # export the model to tensorrt format fp16 and img siae 540
assert success



