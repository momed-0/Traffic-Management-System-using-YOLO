import argparse
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
import publish


# Configuration constants
CONFIG = {
    "video_path": "video/id4.mp4",
    "line_y": 600, # to check if vehicle passed the frame
    "frame_size": (640, 640),
    "publish_interval": 2, # Interval in seconds for publish to cloud
    "time_int": 10.0,      # interval to flush the queue
    "zone_name": "zone1",  
    "model_path": "models/yolov8_yt_model/best.engine",
    "class_list": "config/yolov8_yt_model/class.txt", #class name
    "encoder_model": "deep_sort/networks/mars-small128.pb",
}

# Parse class list dynamically
def parse_class_list(file_path):
    with open(file_path, "r") as file:
        return file.read().strip().split("\n")

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
        self.tracks = None

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


def detect_vehicles(frame, results, class_list):
    detections = {cls: [] for cls in class_list}
    for row in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls_idx = int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4], int(row[5])
        vehicle_type = class_list[cls_idx]
        if vehicle_type in detections:
            detections[vehicle_type].append([x1, y1, x2, y2, score])
    return detections

def parse_arguments():
    """
    Parse command-line arguments, using CONFIG values as defaults.
    """
    parser = argparse.ArgumentParser(description="Traffic Management System")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish data to AWS IoT Core.",
    )
    parser.add_argument(
        "--zone_name",
        type=str,
        default=CONFIG["zone_name"],
        help="Topic name to send MQTT messages.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=CONFIG["video_path"],
        help="Path to the video file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=CONFIG["model_path"],
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--class_list",
        type=str,
        default=CONFIG["class_list"],
        help="Path to the COCO class names file.",
    )
    parser.add_argument(
        "--publish_interval",
        type=int,
        default=CONFIG["publish_interval"],
        help="Interval in publishing messages",
    )
    parser.add_argument(
        "--line_y",
        type=int,
        default=CONFIG["line_y"],
        help="Y Axis coordinate to draw ending line.",
    )
    return parser.parse_args()

def update_config_with_arguments(args):
    """
    Update CONFIG dictionary with command-line argument values.
    """
    for key in CONFIG.keys():
        if hasattr(args, key):
            CONFIG[key] = getattr(args, key)


def main():
    args = parse_arguments()
    update_config_with_arguments(args)

    if args.publish:
        # Establish the connection with AWS IoT
        publish.connect_client(CONFIG["zone_name"])
   
    class_list = parse_class_list(CONFIG["class_list"])
    trackers = {cls: Tracker() for cls in class_list}
    vehicles = {cls: set() for cls in class_list}

    global_id_map = {}
    last_publish_time = time.time()

    model = YOLO(CONFIG["model_path"], task="detect")
    cap = cv2.VideoCapture(CONFIG["video_path"])

    if not cap.isOpened():
        print(f"Error: Unable to open video file {CONFIG['video_path']}.")
        exit()

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame,CONFIG["frame_size"])
        results = model(frame, imgsz=CONFIG["frame_size"][0], verbose=False)
        detections = detect_vehicles(frame, results, class_list)
        det_time = time.time()

        vehicle_updated = False
        for vehicle_type, tracker in trackers.items():
            track_results = tracker.update(frame, detections[vehicle_type])
            for track in track_results:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                vehicle_id = track.track_id
                cy = (y1 + y2) // 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{vehicle_id}: {vehicle_type}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if cy > CONFIG["line_y"]:
                    vehicle_updated = True
                    vehicles[vehicle_type].discard(vehicle_id)
                    if vehicle_id in global_id_map:
                        del global_id_map[vehicle_id]
                else:
                    if vehicle_id not in vehicles[vehicle_type]:
                        vehicle_updated = True
                        vehicles[vehicle_type].add(vehicle_id)
                        global_id_map[vehicle_id] = [det_time,vehicle_type]

        for veh_id, veh_type in list(global_id_map.items()):
            if time.time() - veh_type[0] >= CONFIG["time_int"]:
                vehicle_updated = True
                vehicles[veh_type[1]].discard(veh_id)
                del global_id_map[veh_id]

        cv2.imshow("Traffic-Cloud",frame)

        if vehicle_updated:
            currently_detected_vehicles = {
                "detection_time": int(time.time()),
                "road_name": CONFIG["zone_name"],
            }
            for cls in class_list:
                currently_detected_vehicles[cls] = len(vehicles[cls])
            if args.publish and time.time() - last_publish_time >= CONFIG["publish_interval"]:
                publish.publish_data(currently_detected_vehicles)
                last_publish_time = time.time()  # Reset the timer
            else:
                print(currently_detected_vehicles)
            print('\n'*2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print("Video feed ended!")
    if args.publish:
        publish.disconnect_client()

if __name__ == "__main__":
    main()
