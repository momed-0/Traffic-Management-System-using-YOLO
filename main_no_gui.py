import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import time
from datetime import datetime
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import torch
import publish


# Establish the connection with AWS IoT
publish.connect_client()


# Convert a UNIX timestamp into a human-readable datetime format
def convrt_time_stamp(timestamp):
    return datetime.fromtimestamp(timestamp).isoformat()

# Display detected vehicle data in a readable format
def print_readable_detected_vehicles(detected_vehicles):
    print(f"Timestamp: {detected_vehicles['Timestamp']}")
    print("Detections:")
    for detection in detected_vehicles["detections"]:
        for vehicle_type, count in detection.items():
            print(f"  {vehicle_type}: {count}")
    print()  # Blank line for readability


# Tracker class to handle object tracking using DeepSORT algorithm
class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4  # Controls distance threshold for object matching
        nn_budget = None  # Limits max features for each tracked object
        encoder_model_filename = 'deep_sort/networks/mars-small128.pb'

        # Initialize tracker with cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):
        # Predict and update the tracker when there are no detections
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return self.tracks

        # Prepare bounding boxes and scores for tracker
        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]  # Convert to width and height
        scores = [d[-1] for d in detections]
        features = self.encoder(frame, bboxes)

        # Create Detection objects with bounding box, score, and feature info
        dets = [Detection(bbox, scores[i], features[i]) for i, bbox in enumerate(bboxes)]

        # Predict and update the tracker with new detections
        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()
        return self.tracks

    # Update tracked object list with confirmed tracks only
    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()  # Convert bounding box to top-left, bottom-right format
            id = track.track_id
            tracks.append(Track(id, bbox))

        self.tracks = tracks


# Track object to hold track ID and bounding box information
class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox


torch.cuda.set_device(0)

# Load the YOLO model with TensorRT optimization
model = YOLO("best.engine")

# Capture video from file
cap = cv2.VideoCapture('id4.mp4')
if not cap.isOpened():
    print("Error: Unable to open video file 'id4.mp4'.")
    exit()

# Read class names for the YOLO model to detect specific objects
class_list = []
with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define a vertical line position and offset for detecting crossing vehicles
cy1 = 427
offset = 6

count = 0
# Initialize multiple tracker instances for different vehicle types
tracker = Tracker()
tracker1 = Tracker()
tracker2 = Tracker()
tracker3 = Tracker()

# Lists to store detected vehicle IDs and timestamps
bus, car, auto_rikshaw, motorcycle = [], [], [], []

# Frame processing loop
while True:
    # Continuously read frames from video
    ret, frame = cap.read()
    if not ret:
        break

    # Process every third frame for efficiency
    count += 1
    if count % 3 != 0:
        continue

    # Reduce frame resolution for faster processing
    frame = cv2.resize(frame, (1020, 500))
    results = model(frame, imgsz=640, verbose=False)
    a = results[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    # Separate detections by vehicle type for tracking
    list, list1, list2, list3 = [], [], [], []
    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        score = row[4]
        d = int(row[5])
        c = class_list[d]

        # Organize detections by vehicle type
        if 'bus' in c:
            list.append([x1, y1, x2, y2, score])
        elif 'car' in c:
            list1.append([x1, y1, x2, y2, score])
        elif 'auto-rikshaw' in c:
            list2.append([x1, y1, x2, y2, score])
        elif 'motor-cycle' in c:
            list3.append([x1, y1, x2, y2, score])

    # Update each tracker with relevant bounding boxes
    bbox_idx = tracker.update(frame, list)
    bbox1_idx = tracker1.update(frame, list1)
    bbox2_idx = tracker2.update(frame, list2)
    bbox3_idx = tracker3.update(frame, list3)

    # Track detection crossing the defined vertical line
    detection_found = False

    # Bus tracking
    for track in bbox_idx:
        bbox = track.bbox
        x3, y3, x4, y4 = bbox
        id = track.track_id
        #find the centre of bounding box
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        #check if falls in some offset of the line then we have detected that vehicle
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            #check if its a unique id
            if bus.count(id) == 0:
                #update the detection_found flag
                detection_found = True
                bus.append(id)

    # Car tracking
    for track1 in bbox1_idx:
        bbox1 = track1.bbox
        x5, y5, x6, y6 = bbox1
        id1 = track1.track_id
        cx2 = int(x5 + x6) // 2
        cy2 = int(y5 + y6) // 2
        if cy1 < (cy2 + offset) and cy1 > (cy2 - offset):
            if car.count(id1) == 0:
                #update the detection_found flag
                detection_found = True
                car.append(id1)

    # Auto-rikshaw tracking
    for track2 in bbox2_idx:
        bbox2 = track2.bbox
        x7, y7, x8, y8 = bbox2
        id2 = track2.track_id
        cx3 = int(x7 + x8) // 2
        cy3 = int(y7 + y8) // 2
        if cy1 < (cy3 + offset) and cy1 > (cy3 - offset):
            if auto_rikshaw.count(id2) == 0:
                #update the detection_found flag
                detection_found = True
                auto_rikshaw.append(id2)

    # Motorcycle tracking
    for track3 in bbox3_idx:
        bbox3 = track3.bbox
        x9, y9, x10, y10 = bbox3
        id3 = track3.track_id
        cx4 = int(x9 + x10) // 2
        cy4 = int(y9 + y10) // 2
        if cy1 < (cy4 + offset) and cy1 > (cy4 - offset):
            if motorcycle.count(id3) == 0:
                #update the detection_found flag
                detection_found = True
                motorcycle.append(id3)

    currently_detected_vehicles = {
        "Timestamp": convrt_time_stamp(time.time()),
        "detections": [
            {"car": len(car)},
            {"bus": len(bus)},
            {"auto-rikshaw": len(auto_rikshaw)},
            {"motor-cycle": len(motorcycle)}
        ]
    }
    
     # If new vehicle detected, display count; otherwise, log "No detections"
    if detection_found:
        #print_readable_detected_vehicles(currently_detected_vehicles)
        publish.publish_data(currently_detected_vehicles)
    else:
        print("No detections found at ", convrt_time_stamp(time.time()))
        print()
cap.release()
print("Video feed ended!")
# Disconnect the client once the detection loop is exited
publish.disconnect_client()
