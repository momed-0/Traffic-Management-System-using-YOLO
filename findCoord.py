import cv2

# Callback function to capture mouse click coordinates
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: ({x}, {y})")

video_source = "id4.mp4" 
cap = cv2.VideoCapture(video_source)

# Set up the OpenCV window and bind the mouse callback
cv2.namedWindow("Video Feed")
cv2.setMouseCallback("Video Feed", get_coordinates)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot fetch the frame.")
        break

    # Resize the frame to 640x640
    frame = cv2.resize(frame, (640, 640))

    # Display the resized video frame
    cv2.imshow("Video Feed", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

