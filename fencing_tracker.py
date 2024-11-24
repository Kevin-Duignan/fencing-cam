import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

warnings.simplefilter("ignore")

import cv2
import torch
import numpy as np
from sort.sort import Sort  # From the public SORT repo

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Using the small model for faster detection

# Open video or webcam
cap = cv2.VideoCapture('videoplayback.mp4')  # Replace with 0 for webcam

# Ensure video opens
if not cap.isOpened():
    # print("Error: Could not open video file.")
    exit()

# Initialize SORT tracker
tracker = Sort()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # print("Error: Failed to read frame.")
        break

    # Perform YOLO detection
    results = model(frame)  # Apply YOLOv5 model
    detections = results.xywh[0].cpu().numpy()  # Extract detected boxes (x, y, width, height) as numpy array
    # print("YOLOv5 detections:", detections)  # Debug output to check detections

    boxes = []
    for det in detections:
        confidence = det[4]
        class_id = int(det[5])  # Class ID of the detected object
        if confidence > 0.5 and class_id == 0:  # Class ID 0 is 'person'
            x, y, w, h = map(int, det[:4])  # Convert to integers
            x1, y1 = int(x - w / 2), int(y - h / 2)  # Calculate x1, y1 (top-left)
            x2, y2 = int(x + w / 2), int(y + h / 2)  # Calculate x2, y2 (bottom-right)
            boxes.append([x1, y1, x2, y2, confidence])  # Append box with confidence

    # Perform tracking using SORT
    if len(boxes) > 0:
        np_boxes = np.array(boxes)
        trackers = tracker.update(np_boxes)
        # print("Trackers:", trackers)  # Debug output to check tracker results

        # Store the horizontal midpoints of the boxes for drawing the vertical line
        midpoints_x = []

        # Draw tracking boxes and calculate midpoints
        for track in trackers:
            track_id = int(track[4])  # Extract the track ID
            x1, y1, x2, y2 = map(int, track[:4])

            # Calculate horizontal midpoint for each detected box (ignoring y values)
            x_mid = (x1 + x2) // 2
            midpoints_x.append(x_mid)

            # Draw bounding box and track ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # If we have exactly two midpoints, draw a vertical line between them
        if len(midpoints_x) == 2:
            x1 = midpoints_x[0]  # Horizontal midpoint of the first fencer
            x2 = midpoints_x[1]  # Horizontal midpoint of the second fencer

            # Draw a vertical line at the average x position of both midpoints
            x_midline = (x1 + x2) // 2  # Average of both midpoints to get a line in the middle
            cv2.line(frame, (x_midline, 0), (x_midline, frame.shape[0]), (0, 0, 255), 2)  # Red vertical line

    # Display the frame
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
