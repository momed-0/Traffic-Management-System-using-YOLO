from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")


success = model.export(format="onnx",imgsz=(640,640))  # export the model to onnx 
assert success



