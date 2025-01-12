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


#path for the video feed
VIDEO_PATH = "id4.mp4"
# Define a vertical line position and offset for detecting crossing vehicles
LINE_Y = 610
LINE_X = 610
OFFSET = 6
SIZE_X = 640
SIZE_Y=640
TIME_INT = 2.0

# Convert a UNIX timestamp into a human-readable datetime format
def convrt_time_stamp(timestamp):
    return datetime.fromtimestamp(timestamp).isoformat()

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
model = YOLO("best.engine",task="detect")

# trying a gstreamer pipeline - change it for video source
pipeline = f"filesrc location={VIDEO_PATH} ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,width={SIZE_X},height={SIZE_Y} ! appsink"


# Capture video from file
cap = cv2.VideoCapture(VIDEO_PATH)
#cap = cv2.VideoCapture(pipeline,cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to open video file 'id4.mp4'.")
    print(cap)
    exit()

# Read class names for the YOLO model to detect specific objects
class_list = ["motor-cycle","car","auto-rikshaw","bus"]

# with open("coco1.txt", "r") as my_file:
#    class_list = my_file.read().split('\n')

count = 0

#Initialize multiple tracker instances for different vehicle types
tracker = Tracker()
tracker1 = Tracker()
tracker2 = Tracker()
tracker3 = Tracker()

# global Lists to store detected vehicle IDs and timestamps
#global structure to hold the unique vehicles count
vehicles = {
    "bus": set(),
    "car": set(),
    "auto-rikshaw": set(),
    "motor-cycle": set()
}

global_id_map = {}

# Frame processing loop
while True:
    # Continuously read frames from video
    ret, frame = cap.read()    
    if not ret:
        break
    #conv RGBA to RGB
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Process every third frame for efficiency
    count += 1
    if count % 3 != 0:
        continue

    # Reduce frame resolution for faster processing
    frame = cv2.resize(frame, (SIZE_X, SIZE_Y))
    results = model(frame,verbose=False)
    a = results[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    det_time = time.time()
    # Separate detections by vehicle type for tracking
    list_bus, list_car, list_auto, list_motor = [], [], [], []
    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        score = row[4]
        d = int(row[5])
        c = class_list[d]

        # Organize detections by vehicle type
        if 'bus' in c:
            list_bus.append([x1, y1, x2, y2, score])
        elif 'car' in c:
            list_car.append([x1, y1, x2, y2, score])

        elif 'auto-rikshaw' in c:
            list_auto.append([x1, y1, x2, y2, score])

        elif 'motor-cycle' in c:
            list_motor.append([x1, y1, x2, y2, score])


    #Update each tracker with relevant bounding boxes
    bbox_idx = tracker.update(frame, list_bus)
    bbox1_idx = tracker1.update(frame, list_car)
    bbox2_idx = tracker2.update(frame, list_auto)
    bbox3_idx = tracker3.update(frame, list_motor)

    #Track detection crossing the defined vertical line
    did_update = False

    #Bus tracking
    for track in bbox_idx:
       bbox = track.bbox
       x3, y3, x4, y4 = bbox
       bus_id = track.track_id
       #find the centre of bounding box
       cx = int(x3 + x4) // 2
       cy = int(y3 + y4) // 2
       cv2.rectangle(frame,(int(x3),int(y3)),(int(x4),int(y4)) ,(0, 255, 0), 2)
       #check if falls in some offset of the line then we have detected that vehicle
       cv2.putText(frame, f"{bus_id}: Bus", (int(x3), int(y3) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       if cy > LINE_Y:
            did_update = True
            vehicles["bus"].discard(bus_id)
            if bus_id in global_id_map:
                del global_id_map[bus_id]
       else:
            if bus_id not in vehicles["bus"]:
                did_update = True
                vehicles["bus"].add(bus_id)
                global_id_map[bus_id] = [det_time, "car"] 
    #Car tracking
    for track1 in bbox1_idx:
       bbox1 = track1.bbox
       x5, y5, x6, y6 = bbox1
       id1 = track1.track_id
       cx2 = int(x5 + x6) // 2
       cy2 = int(y5 + y6) // 2
       cv2.rectangle(frame, (int(x5),int(y5)), (int(x6),int(y6)),(0,255,0),2)
       cv2.putText(frame,f"{id1}: Car",(int(x5),int(y5)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
       if cy2 > LINE_Y:
            did_update = True
            vehicles["car"].discard(id1)
            if id1 in global_id_map:
                del global_id_map[id1]

       else:
            if id1 not in vehicles["car"]:
                did_update = True
                vehicles["car"].add(id1)
                global_id_map[id1] = [det_time,"car"]
    
    # Auto-rikshaw tracking
    for track2 in bbox2_idx:
        bbox2 = track2.bbox
        x7, y7, x8, y8 = bbox2
        id2 = track2.track_id
        cx3 = int(x7 + x8) // 2
        cy3 = int(y7 + y8) // 2
        cv2.rectangle(frame, (int(x7),int(y7)), (int(x8),int(y8)),(0,255,0),2)
        cv2.putText(frame,f"{id2}: Auto",(int(x7),int(y7)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        if cy3 > LINE_Y:
            did_update = True
            vehicles["auto-rikshaw"].discard(id2)
            if id2 in global_id_map:
                del global_id_map[id2]
        else:
            if id2 not in vehicles["auto-rikshaw"]:
                did_update = True
                vehicles["auto-rikshaw"].add(id2)
                global_id_map[id2] = [det_time,"auto-rikshaw"]
    # Motorcycle tracking
    for track3 in bbox3_idx:
        bbox3 = track3.bbox
        x9, y9, x10, y10 = bbox3
        id3 = track3.track_id
        cx4 = int(x9 + x10) // 2
        cy4 = int(y9 + y10) // 2
        cv2.rectangle(frame, (int(x9),int(y9)), (int(x10),int(y10)),(0,255,0),2)
        cv2.putText(frame,f"{id3}: Bike",(int(x9),int(y9)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        if cy4 > LINE_Y:
            did_update = True
            vehicles["motor-cycle"].discard(id3)
            if id3 in global_id_map:
                del global_id_map[id3]
        else:
            if id3 not in vehicles["motor-cycle"]:
                did_update = True
                vehicles["motor-cycle"].add(id3)
                global_id_map[id3] = [det_time,"motor-cycle"]
    for veh_id, entries in list(global_id_map.items()):
        if det_time - entries[0] >= TIME_INT:
            did_update = True
            vehicles[entries[1]].discard(veh_id)
            del global_id_map[veh_id]
    cv2.line(frame,(0,LINE_Y),(640,LINE_Y),(255,255,255),2)
    cv2.imshow("Traffic-Cloud",frame)
    if did_update:
        #detected vehicles in this frame    
        currently_detected_vehicles = {
            "Timestamp": convrt_time_stamp(time.time()),
            "detections": [
                {"car": len(vehicles["car"])},
                {"bus": len(vehicles["bus"])},
                {"auto-rikshaw": len(vehicles["auto-rikshaw"])},
                {"motor-cycle": len(vehicles["motor-cycle"])}
            ]
        }
        #Current State of Vehicles on Screen"
        print(currently_detected_vehicles)
        print('\n'*2)
    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
print("Video feed ended!")

