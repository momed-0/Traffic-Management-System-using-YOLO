import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import time
import gc
import math
import cvzone
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection


import torch

#optimized tracker class
class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'deep_sort/networks/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return self.tracks

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()
        return self.tracks
    

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            id = track.track_id

            tracks.append(Track(id, bbox))

        self.tracks = tracks

class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox


torch.cuda.set_device(0)

model=YOLO("best.engine",task="detect") #tensorrt optimized model

cap=cv2.VideoCapture('id4.mp4')

if not cap.isOpened():
    print("Error: Unable to open video file 'id4.mp4' .")
    exit()

class_list = []

with open("coco1.txt","r") as my_file:
    class_list = my_file.read().split("\n")

cy1=427
offset=6

count=0
tracker=Tracker()
tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()
bus=[]
car=[]
auto_rikshaw=[]
motorcycle=[]
while True:    
    ret,frame = cap.read()
    if not ret:
       break
   # print(f"Frame size: {frame.shape}")  # prints the size of the frame
 

    count += 1
    if count % 3 != 0: 
        continue
    #frame=cv2.resize(frame,(640, 640)) #reduce the resolution
    #results=model.predict(frame)
    results = model(frame, imgsz=640)
    a = results[0].boxes.data
    a_cpu = a.cpu().numpy()
    px = pd.DataFrame(a_cpu).astype("float")
    #print(px)
    
    list=[]
    list1=[]
    list2=[]
    list3=[]
    
    for index,row in px.iterrows():
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        score=row[4]
        d=int(row[5])
        c=class_list[d]
       
        if 'bus' in c:
            list.append([x1,y1,x2,y2,score])
        elif 'car' in c:
             list1.append([x1,y1,x2,y2,score])
        elif 'auto-rikshaw' in c:
             list2.append([x1,y1,x2,y2,score])
        elif 'motor-cycle' in c:
             list3.append([x1,y1,x2,y2,score])
        
    bbox_idx=tracker.update(frame,list)
    bbox1_idx=tracker1.update(frame,list1)
    bbox2_idx=tracker2.update(frame,list2)
    bbox3_idx=tracker3.update(frame,list3)
  
    for track in bbox_idx:
        bbox=track.bbox
        x3,y3,x4,y4=bbox
        id=track.track_id
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if cy1<(cy+offset) and cy1>(cy-offset):
           cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
           cvzone.putTextRect(frame, f'{id}', (int(x3), int(y3)), 1, 1)

           time_bus = time.time()
           if bus.count(id)==0:
              bus.append((id,time_bus))
              print("BUS",(len(bus)), "at ",time_bus )
#CAR
    for track1 in bbox1_idx:
        bbox1=track1.bbox
        x5,y5,x6,y6=bbox1
        id1=track1.track_id
        cx2=int(x5+x6)//2
        cy2=int(y5+y6)//2
        if cy1<(cy2+offset) and cy1>(cy2-offset):
           cv2.rectangle(frame, (int(x5), int(y5)), (int(x6), int(y6)), (0, 255, 0), 2)
           cvzone.putTextRect(frame, f'{id1}', (int(x5), int(y5)), 1, 1)
           
           time_car = time.time()
           if car.count(id1)==0:
              car.append((id1,time_car))
              print("CAR",(len(car)),"at ",time_car)
#auto-rikshaw
    for track2 in bbox2_idx:
        bbox2=track2.bbox
        x7,y7,x8,y8=bbox2
        id2=track2.track_id
        cx3=int(x7+x8)//2
        cy3=int(y7+y8)//2
        if cy1<(cy3+offset) and cy1>(cy3-offset):
           cv2.rectangle(frame, (int(x7), int(y7)), (int(x8), int(y8)), (0, 255, 0), 2)
           cvzone.putTextRect(frame,f'{id2}',(int(x7),int(y7)),1,1)
           
           time_auto = time.time()
           if auto_rikshaw.count(id2)==0:
              auto_rikshaw.append((id2,time_auto))
              print("AUTO_RIKSHAW",(len(auto_rikshaw)),"at ",time_auto)
#motorcycle
    for track3 in bbox3_idx:
        bbox3=track3.bbox
        x9,y9,x10,y10=bbox3
        id3=track3.track_id
        cx4=int(x9+x10)//2
        cy4=int(y9+y10)//2
        if cy1<(cy4+offset) and cy1>(cy4-offset):
           cv2.rectangle(frame, (int(x9), int(y9)), (int(x10), int(y10)), (0, 255, 0), 2)
           cvzone.putTextRect(frame, f'{id3}', (int(x9), int(y9)), 1, 1)
          # cv2.putText(frame, f'{id3}', (int(x9),int(y9)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
           time_motor = time.time()
           if motorcycle.count(id3)==0:
              motorcycle.append((id3,time_motor))
              print("MOTORCYCLE",(len(motorcycle)),"at ",time_motor)          
    countbus=(len(bus))
    countcar=(len(car))
    countauto_rikshaw=(len(auto_rikshaw))
    countmotorcycle=(len(motorcycle))
    #cv2.putText(frame, f'Auto_Rikshaw: {countauto_rikshaw}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #cv2.putText(frame, f'Bus: {countbus}', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #cv2.putText(frame, f'Car: {countcar}', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
   # cv2.putText(frame, f'Motorcycle: {countmotorcycle}', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cvzone.putTextRect(frame,f'Auto_Rikshaw : {countauto_rikshaw}',(50,60),scale=2,thickness=2,colorR=(0,0,255))
    cvzone.putTextRect(frame,f'Bus : {countbus}',(50,120),scale=2,thickness=2,colorR=(0,0,255))
    cvzone.putTextRect(frame,f'Car : {countcar}',(50,180),scale=2,thickness=2,colorR=(0,0,255))
    cvzone.putTextRect(frame,f'MotorCycle : {countmotorcycle}',(50,240),scale=2,thickness=2,colorR=(0,0,255))


    cv2.line(frame,(328,226),(408,226),(255,255,255),2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

    gc.collect()
cap.release()    
cv2.destroyAllWindows()




