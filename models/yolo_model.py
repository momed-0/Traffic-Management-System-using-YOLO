from ultralytics import YOLO
import torch

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        torch.cuda.set_device(0)

    def predict(self, frame):
        results = self.model.predict(frame)
        return results[0].boxes.data.cpu().numpy()
