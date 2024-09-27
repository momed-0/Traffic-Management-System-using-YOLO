import cv2
from models.yolo_model import YOLOModel
from trackers.vehicle_tracker import Tracker
from utils.video_utils import draw_boxes, display_vehicle_count, count_vehicles, categorize_detections
from config.config import VIDEO_PATH, CLASS_FILE, CY1, OFFSET

def process_video(video_path, model, class_list, trackers, line_y, offset):
    cap = cv2.VideoCapture(video_path)  # Open video file
    vehicle_ids = {vtype: [] for vtype in trackers.keys()}  # To keep track of counted vehicle IDs

    while cap.isOpened():
        ret, frame = cap.read()  # Read each frame
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))  # Resize frame for consistent processing
        detections = model.predict(frame)  # Get detections from YOLO
        detection_by_class = categorize_detections(detections, class_list)  # Sort detections by vehicle type

        # Update trackers and draw boxes for each vehicle type
        for vehicle_type, tracker in trackers.items():
            tracks = tracker.update(frame, detection_by_class[vehicle_type])
            draw_boxes(frame, tracks, label=vehicle_type)
            count_vehicles(vehicle_ids[vehicle_type], tracks, line_y, offset)

        # Display the vehicle counts
        display_vehicle_count(frame, {vtype: len(ids) for vtype, ids in vehicle_ids.items()})

        # Draw the counting line
        cv2.line(frame, (405, line_y), (580, line_y), (255, 255, 255), 2)
        cv2.imshow("Traffic Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC'
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    class_list = open(CLASS_FILE).read().strip().split('\n')  # Load COCO class names
    yolo_model = YOLOModel(MODEL_PATH)  # Initialize YOLO model

    trackers = {
        "bus": Tracker(),
        "car": Tracker(),
        "auto-rikshaw": Tracker(),
        "motorcycle": Tracker()
    }

    process_video(VIDEO_PATH, yolo_model, class_list, trackers, CY1, OFFSET)
