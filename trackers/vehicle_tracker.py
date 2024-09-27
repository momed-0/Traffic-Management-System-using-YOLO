from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np
from config.config import ENCODER_MODEL_PATH, MAX_COSINE_DISTANCE, NN_BUDGET

class Tracker:
    def __init__(self):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
        self.tracker = DeepSortTracker(metric)  # Initialize DeepSort tracker with metric
        self.encoder = gdet.create_box_encoder(ENCODER_MODEL_PATH, batch_size=1)  # Encode bounding boxes for tracking

    def update(self, frame, detections):
        # If no detections, just predict the next position
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            return []

        # Extract bounding boxes and features from detections
        bboxes = np.asarray([d[:-1] for d in detections])  # Bounding boxes
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]  # Convert bbox format
        scores = [d[-1] for d in detections]  # Detection scores

        features = self.encoder(frame, bboxes)  # Generate features for detections
        dets = [Detection(bbox, scores[i], features[i]) for i, bbox in enumerate(bboxes)]  # Wrap detections

        self.tracker.predict()
        self.tracker.update(dets)  # Update tracker with new detections

        return [Track(track.track_id, track.to_tlbr()) for track in self.tracker.tracks if track.is_confirmed()]

class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id  # Unique ID for each track
        self.bbox = bbox  # Bounding box (x1, y1, x2, y2)
