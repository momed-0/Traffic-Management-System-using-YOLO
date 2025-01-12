import cv2
import pandas as pd
from ultralytics import YOLO
import time
from datetime import datetime
import torch
import math

#path for the video feed
VIDEO_PATH = 'id4.mp4'
# Define a vertical line position and offset for detecting crossing vehicles
LINE_Y = 610
OFFSET = 5
SIZE_X = 640
SIZE_Y=640
TIME_INT = 5.0

# Convert a UNIX timestamp into a human-readable datetime format
def convrt_time_stamp(timestamp):
    return datetime.fromtimestamp(timestamp).isoformat()


class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
        # center point of object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


torch.cuda.set_device(0)

# Load the YOLO model with TensorRT optimization
model = YOLO("best.engine", task="detect")

# trying a gstreamer pipeline - change it for video source
pipeline = f"filesrc location={VIDEO_PATH} ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,width={SIZE_X},height={SIZE_Y} ! appsink"


# Capture video from file
#cap = cv2.VideoCapture(VIDEO_PATH)
cap = cv2.VideoCapture(pipeline,cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to open video file 'id4.mp4'.")
    exit()

# Read class names for the YOLO model to detect specific objects
class_list = ["motor-cycle","car","auto-rikshaw","bus"]

# with open("coco1.txt", "r") as my_file:
#    class_list = my_file.read().split('\n')


tracker=Tracker()
tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()

#global structure to hold the unique vehicles
vehicles = {
    "bus": set(),
    "car": set(),
    "auto-rikshaw": set(),
    "motor-cycle": set()
}

global_id_map = {}

count = 0 
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
    #frame = cv2.resize(frame, (SIZE_X, SIZE_Y))
    results = model(frame, imgsz=640,verbose=False)
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
            list_bus.append([x1, y1, x2, y2])
        elif 'car' in c:
            list_car.append([x1, y1, x2, y2])

        elif 'auto-rikshaw' in c:
            list_auto.append([x1, y1, x2, y2])

        elif 'motor-cycle' in c:
            list_motor.append([x1, y1, x2, y2])
    #whenever the vehicle passes through the frame , remove it from global list
    did_update = False
    bus_tracker = tracker.update(list_bus)
    car_tracker = tracker1.update(list_car)
    auto_tracker = tracker2.update(list_auto)
    motor_tracker = tracker3.update(list_motor)

    #check if it crossed the line , if then remove it from list, 
    #else check if its a unique id , then add it into list
    for bbox in bus_tracker:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if LINE_Y < (cy + OFFSET) and LINE_Y > (cy - OFFSET):
            did_update = True
            #remove the occurence if it exists in global list
            print("A Bus left the frame!")
            vehicles["bus"].discard(id)
            if id in global_id_map:
                del global_id_map[id]
        else:
            #add to the list if we haven't detected this earlier
            if id not in vehicles["bus"]:
                did_update = True
                vehicles["bus"].add(id)
                global_id_map[id] = [det_time,"bus"]
    #CAR
    for bbox1 in car_tracker:
        x5,y5,x6,y6,id1=bbox1
        cx2=int(x5+x6)//2
        cy2=int(y5+y6)//2
        if LINE_Y < (cy2 + OFFSET) and LINE_Y > (cy2 - OFFSET):
            did_update = True
            print("A Car left the frame!") 
            vehicles["car"].discard(id1)
            if id1 in global_id_map:
                del global_id_map[id1]
        else:
           if id1 not in vehicles["car"]:
                did_update = True
                vehicles["car"].add(id1)
                global_id_map[id1] = [det_time,"car"]
    #auto-rikshaw
    for bbox2 in auto_tracker:
        x7,y7,x8,y8,id2=bbox2
        cx3=int(x7+x8)//2
        cy3=int(y7+y8)//2
        if LINE_Y < (cy3 + OFFSET) and LINE_Y > (cy3 - OFFSET):
            did_update = True
            print("A Auto-Rikshaw left the frame!") 
            vehicles["auto-rikshaw"].discard(id2)
            if id2 in global_id_map:
                del global_id_map[id2]
        else:
           if id2 not in vehicles["auto-rikshaw"]:
                did_update = True
                vehicles["auto-rikshaw"].add(id2)
                global_id_map[id2] = [det_time,"auto-rikshaw"]
    #motorcycle
    for bbox3 in motor_tracker:
        x9,y9,x10,y10,id3=bbox3
        cx4=int(x9+x10)//2
        cy4=int(y9+y10)//2
        if LINE_Y < (cy4 + OFFSET) and LINE_Y  > (cy4 - OFFSET):
            did_update = True
            print("A Motor-Cycle left the frame!") 
            vehicles["motor-cycle"].discard(id3)
            if id3 in global_id_map:
                del global_id_map[id3]
        else:
           if id3 not in vehicles["motor-cycle"]:
                did_update = True
                vehicles["motor-cycle"].add(id3)
                global_id_map[id3] = [det_time,"motor-cycle"]
    # traverse through the global id map and flush the items after certain time
    for veh_id, entries in list(global_id_map.items()):
        if det_time - entries[0] >= TIME_INT:
            did_update = True
            vehicles[entries[1]].discard(veh_id)
            del global_id_map[veh_id]

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

        print('\n' * 2)

cap.release()
print("Video feed ended!")
