import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import time
from datetime import datetime
import torch


#path for the video feed
VIDEO_PATH = 'id4.mp4'
# Define a vertical line position and offset for detecting crossing vehicles
LINE_Y = 637
OFFSET = 6
SIZE_X = 640
SIZE_Y=640

# Convert a UNIX timestamp into a human-readable datetime format
def convrt_time_stamp(timestamp):
    return datetime.fromtimestamp(timestamp).isoformat()

torch.cuda.set_device(0)

# Load the YOLO model with TensorRT optimization
model = YOLO("best.engine")

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


#global structure to hold the unique vehicles count
vehicles = {
    "bus": 0,
    "car": 0,
    "auto-rikshaw": 0,
    "motor-cycle": 0
}

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
    if count % 6 != 0:
        continue

    # Reduce frame resolution for faster processing
    #frame = cv2.resize(frame, (SIZE_X, SIZE_Y))
    results = model(frame, imgsz=640, verbose=False)
    a = results[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    did_update = False
    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        #score = row[4]
        d = int(row[5])
        c = class_list[d]
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2

        #check if the vehicle has crossed the line , then remove else don't
        if LINE_Y < (cy + OFFSET) and LINE_Y > (cy - OFFSET):
            #decrement the count
            if vehicles[c] > 0 :
                did_update = True
                vehicles[c] -= 1
                print()
                print(f"Deleted {c} from global")
                print()
        else:
            #increment the list
            vehicles[c] += 1
            did_update = True
    
    if did_update:
        #detected vehicles in this frame    
        currently_detected_vehicles = {
            "Timestamp": convrt_time_stamp(time.time()),
            "detections": [
                {"car": vehicles["car"]},
                {"bus": vehicles["bus"]},
                {"auto-rikshaw": vehicles["auto-rikshaw"]},
                {"motor-cycle": vehicles["motor-cycle"]}
            ]
        }
        #Current State of Vehicles on Screen"
        print(currently_detected_vehicles)

cap.release()
print("Video feed ended!")
