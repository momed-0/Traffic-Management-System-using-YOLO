import cv2
import cvzone

def draw_boxes(frame, tracks, color=(0, 255, 0), label="ID"):
    for track in tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cvzone.putTextRect(frame, f'{track.track_id}', (x1, y1), 1, 1)

def display_vehicle_count(frame, vehicle_counts):
    for i, (vehicle_type, count) in enumerate(vehicle_counts.items()):
        cvzone.putTextRect(frame, f'{vehicle_type}: {count}', (50, 60 + 60 * i), scale=2, thickness=2, colorR=(0, 0, 255))

def count_vehicles(vehicle_ids, tracks, line_y, offset):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if line_y - offset < cy < line_y + offset and track.track_id not in vehicle_ids:
            vehicle_ids.append(track.track_id)

def categorize_detections(detections, class_list):
    vehicle_detections = {"bus": [], "car": [], "auto-rikshaw": [], "motorcycle": []}
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        vehicle_type = class_list[int(class_id)]
        if vehicle_type in vehicle_detections:
            vehicle_detections[vehicle_type].append([x1, y1, x2, y2, score])
    return vehicle_detections
