import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.simplefilter("ignore")

import cv2
import torch
import numpy as np
from sort.sort import Sort
import socket
import time

# --- WiFi TCP connection to ESP32 ---
ESP32_HOST = "esp32.local"  # mDNS hostname
ESP32_PORT = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# Retry until connection succeeds
while True:
    try:
        print("Connecting to ESP32...")
        s.connect((ESP32_HOST, ESP32_PORT))
        print("✅ Connected to ESP32 over WiFi")
        break
    except Exception as e:
        print("❌ Connection failed, retrying in 2s:", e)
        time.sleep(2)

# --- Load YOLOv5 ---
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open video source")
    exit()

tracker = Sort()

# --- Servo Control Parameters ---
max_angle = 180
min_angle = 0
k_p = 0.2
k_d = 0.05
deadzone = 10
max_step = 5
current_servo_angle = 90
previous_error = 0

# --- Helper functions ---
def process_trackers(frame, trackers):
    midpoints_x = []
    for track in trackers:
        track_id = int(track[4])
        x1, y1, x2, y2 = map(int, track[:4])
        x_mid = (x1 + x2) // 2
        midpoints_x.append(x_mid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return midpoints_x

def update_servo(target_x, frame_width):
    global current_servo_angle, previous_error
    frame_center = frame_width // 2
    error = frame_center - target_x

    if abs(error) > deadzone:
        derivative = error - previous_error
        delta_angle = int(k_p * error + k_d * derivative)
        delta_angle = max(-max_step, min(max_step, delta_angle))

        current_servo_angle += delta_angle
        current_servo_angle = max(min_angle, min(max_angle, current_servo_angle))

        # Send angle to ESP32
        s.sendall(f"{current_servo_angle}\n".encode())

    previous_error = error

# --- Tracking functions ---
def track_single_person():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xywh[0].cpu().numpy()
        boxes = []
        for det in detections:
            confidence = det[4]
            class_id = int(det[5])
            if confidence > 0.5 and class_id == 0:  # Person
                x, y, w, h = map(int, det[:4])
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                boxes.append([x1, y1, x2, y2, confidence])

        if boxes:
            trackers_array = tracker.update(np.array(boxes))
            midpoints_x = process_trackers(frame, trackers_array)
            if midpoints_x:
                x_mid = midpoints_x[0]
                update_servo(x_mid, frame.shape[1])

        cv2.imshow("Single Person Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def track_fencers():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xywh[0].cpu().numpy()
        boxes = []
        for det in detections:
            confidence = det[4]
            class_id = int(det[5])
            if confidence > 0.5 and class_id == 0:
                x, y, w, h = map(int, det[:4])
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                boxes.append([x1, y1, x2, y2, confidence])

        if boxes:
            trackers_array = tracker.update(np.array(boxes))
            midpoints_x = process_trackers(frame, trackers_array)
            if len(midpoints_x) >= 2:
                x1, x2 = midpoints_x[:2]
                x_midline = (x1 + x2) // 2
                update_servo(x_midline, frame.shape[1])

        cv2.imshow("Fencers Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Main ---
if __name__ == "__main__":
    mode = input("Select mode: [1] Single Person, [2] Fencers: ")
    if mode == "1":
        track_single_person()
    elif mode == "2":
        track_fencers()
    else:
        print("Invalid selection")
