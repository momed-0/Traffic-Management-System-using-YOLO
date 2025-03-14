import argparse
import yaml
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
    "video_path": "video/test.mp4",
    "line_y": 600, # to check if vehicle passed the frame
    "frame_size": (640, 640),
    "publish_interval": 2, # Interval in seconds for publish to cloud
    "zone_name": "zone1",  
    "model_path": "models/yolov11n/yolov11n.engine",
    "class_list": "config/yolov11n/class.txt", #class name
    "encoder_model": "deep_sort/networks/mars-small128.pb",
    "auto_model":   "models/auto.engine",
    "full_class":   "config/yolov11n/full_class.yml",
    }

# Parse class list dynamically
def parse_class_list(file_path):
    with open(file_path, "r") as file:
        return file.read().strip().split("\n")

def parse_full_class_list(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

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


def detect_vehicles(frame, results, class_list, full_class_list):
    detections = {cls: [] for cls in class_list}
    for row in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls_idx = int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4], int(row[5])
        vehicle_type = full_class_list[cls_idx] 
        if vehicle_type in class_list:   # filter the result based on the requirements
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
        "--full_class",
        type=str,
        default=CONFIG["full_class"],
        help="Path to YAML file of class names",
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

def print_count_of_vehicles(vehicles, args, publish, last_publish_time):
    """
    Print the no of vehicles in each type
    """
    currently_detected_vehicles = {
                "detection_time": int(time.time()),
                "road_name": CONFIG["zone_name"],
    }
    currently_detected_vehicles.update({cls: len(vehicles[cls]) for cls in vehicles})

    if args.publish and time.time() - last_publish_time >= CONFIG["publish_interval"]:
        publish.publish_data(currently_detected_vehicles)
        last_publish_time = time.time()  # Reset the timer
    else:
        print(currently_detected_vehicles)
        print('\n'*2)
    return last_publish_time  # Return the updated value

def print_ids_of_vehicles(vehicles, args, publish, last_publish_time):
    """
    Print the no of ids and details of each stored vehicles
    """
    currently_detected_vehicles = {
                "detection_time": int(time.time()),
                "road_name": CONFIG["zone_name"],
    }
    currently_detected_vehicles.update({cls: vehicles[cls] for cls in vehicles})

    if args.publish and time.time() - last_publish_time >= CONFIG["publish_interval"]:
        publish.publish_data(currently_detected_vehicles)
        last_publish_time = time.time()  # Reset the timer
    else:
        print(currently_detected_vehicles)
        print('\n'*2)
    return last_publish_time  # Return the updated value


def main():
    args = parse_arguments()
    update_config_with_arguments(args)

    if args.publish:
        # Establish the connection with AWS IoT
        publish.connect_client(CONFIG["zone_name"])
   
    class_list = parse_class_list(CONFIG["class_list"])
    full_class_list=parse_full_class_list(CONFIG["full_class"])

    trackers = {cls: Tracker() for cls in class_list}
    vehicles = {cls: {} for cls in class_list}

    last_publish_time = time.time()

    model = YOLO(CONFIG["model_path"], task="detect")
    auto_model = YOLO(CONFIG["auto_model"], task="detect")

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
        results_auto = auto_model(frame, imgsz=CONFIG["frame_size"][0], verbose=False)

        detections = detect_vehicles(frame, results, class_list, full_class_list)
        
        detections_auto = detect_vehicles(frame, results_auto,class_list, {0:"Auto"})
        detections["Auto"] = detections_auto["Auto"]
        det_time = time.time()

        # dictionary for each vehicle type mapping id -> detection parameters
        vehicles = {cls: {} for cls in class_list}

        for vehicle_type, tracker in trackers.items():
            track_results = tracker.update(frame, detections[vehicle_type])
            for track in track_results:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                vehicle_id = track.track_id
                cy = (y1 + y2) // 2
                cx = (x1 + x2) // 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{vehicle_id}: {vehicle_type}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # insert the result in dictionary
                vehicles[vehicle_type][vehicle_id] = {"cx": cx, "cy": cy }
        
        cv2.imshow("Traffic-Cloud",frame)

        last_publish_time = print_count_of_vehicles(vehicles, args, publish, last_publish_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print("Video feed ended!")
    if args.publish:
        publish.disconnect_client()

if __name__ == "__main__":
    main()
