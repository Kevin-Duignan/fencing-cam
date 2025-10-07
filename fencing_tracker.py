import cv2
import torch
import numpy as np
import socket
import time
import warnings
from collections import deque
from threading import Thread, Lock
import queue

# -----------------------------
# Suppress warnings
# -----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*")

# -----------------------------
# Device setup
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -----------------------------
# Load YOLOv5n model with optimizations
# -----------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.to(device)
model.eval()

# Optimize model
if device == 'cuda':
    model.half()  # FP16 for speed
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions

# Compile model for faster inference (PyTorch 2.0+)
# Disabled on Windows due to C++ compiler requirements
# try:
#     model = torch.compile(model, mode='reduce-overhead')
#     print("✅ Model compiled with torch.compile")
# except:
#     print("ℹ️ torch.compile not available, using standard model")

# -----------------------------
# Resolve ESP32 hostname via mDNS
# -----------------------------
ESP32_HOSTNAME = "esp32.local"
ESP32_PORT = 12345

def resolve_mdns(hostname, port):
    try:
        ip = socket.gethostbyname(hostname)
        return ip, port
    except Exception as e:
        print(f"⚠️ Could not resolve {hostname}: {e}")
        return None, None

ESP32_IP, ESP32_PORT = resolve_mdns(ESP32_HOSTNAME, ESP32_PORT)
if ESP32_IP is None:
    print("❌ Could not resolve ESP32 hostname. Make sure your PC is on the same WiFi as ESP32.")
    exit()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
try:
    sock.connect((ESP32_IP, ESP32_PORT))
    print(f"✅ Connected to ESP32 at {ESP32_IP}:{ESP32_PORT}")
except Exception as e:
    print("❌ Could not connect to ESP32:", e)
    exit()

# -----------------------------
# Camera setup with threading
# -----------------------------
class VideoCapture:
    """Threaded video capture for reduced latency"""
    def __init__(self, src=0, backend=cv2.CAP_MSMF):
        self.cap = cv2.VideoCapture(src, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.q = queue.Queue(maxsize=2)
        self.lock = Lock()
        self.stopped = False
        
        # Warm up
        for _ in range(5):
            self.cap.read()
        
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()
    
    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Keep only latest frame
            with self.lock:
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)
    
    def read(self):
        return self.q.get()
    
    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
    
    def isOpened(self):
        return self.cap.isOpened()

# Initialize camera
cap = None
camera_indices = [2, 1, 0]
backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_VFW]

for idx in camera_indices:
    for backend in backends:
        try:
            cap_try = VideoCapture(idx, backend)
            if cap_try.isOpened():
                cap = cap_try
                print(f"✅ Using camera index {idx} with backend {backend}")
                break
        except:
            continue
    if cap is not None:
        break

if cap is None:
    print("❌ Could not open any camera. Make sure Iriun is running and connected.")
    exit()

time.sleep(0.5)

# -----------------------------
# Servo PID with Kalman filter
# -----------------------------
class ServoController:
    def __init__(self):
        self.current_angle = 90
        self.previous_error = 0
        self.integral = 0
        self.history = deque(maxlen=5)
        
        # PID gains
        self.k_p = 0.20
        self.k_i = 0.005
        self.k_d = 0.08
        self.deadzone = 8
        self.max_step = 6
        
        # Simple Kalman filter for position smoothing
        self.kalman_gain = 0.3
        self.estimated_pos = None
        self.estimation_error = 1.0
        
        self.last_command_time = 0
        self.min_command_interval = 0.03  # 30ms between commands
    
    def kalman_update(self, measurement):
        """Simple 1D Kalman filter"""
        if self.estimated_pos is None:
            self.estimated_pos = measurement
            return measurement
        
        # Prediction (assume constant position)
        predicted_error = self.estimation_error + 0.1
        
        # Update
        kalman_gain = predicted_error / (predicted_error + 5.0)
        self.estimated_pos = self.estimated_pos + kalman_gain * (measurement - self.estimated_pos)
        self.estimation_error = (1 - kalman_gain) * predicted_error
        
        return self.estimated_pos
    
    def update(self, target_x, frame_width):
        # Apply Kalman filter
        filtered_x = self.kalman_update(target_x)
        
        # Moving average
        self.history.append(filtered_x)
        smoothed_target = int(np.mean(self.history))
        
        # PID control
        error = frame_width // 2 - smoothed_target
        
        if abs(error) > self.deadzone:
            self.integral = np.clip(self.integral + error, -100, 100)
            derivative = error - self.previous_error
            
            delta = int(self.k_p * error + self.k_i * self.integral + self.k_d * derivative)
            delta = np.clip(delta, -self.max_step, self.max_step)
            
            new_angle = np.clip(self.current_angle + delta, 0, 180)
            
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_command_time >= self.min_command_interval:
                if abs(new_angle - self.current_angle) > 0:
                    try:
                        sock.sendall(f"{new_angle}\n".encode())
                        self.current_angle = new_angle
                        self.last_command_time = current_time
                    except Exception as e:
                        print("⚠️ ESP32 connection lost:", e)
        else:
            self.integral *= 0.9  # Decay integral when in deadzone
        
        self.previous_error = error

servo_controller = ServoController()

# -----------------------------
# Optimized tracking functions
# -----------------------------
class FPSCounter:
    def __init__(self, window=30):
        self.timestamps = deque(maxlen=window)
    
    def update(self):
        self.timestamps.append(time.time())
        if len(self.timestamps) > 1:
            return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])
        return 0

def preprocess_frame(frame, target_size=(320, 320)):
    """Efficient preprocessing with caching"""
    # Resize once to target size
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    # Convert color space
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb

def track_single_person(target_fps=30):
    fps_counter = FPSCounter()
    last_inference_time = 0
    inference_interval = 1.0 / 15  # Run inference at 15 Hz max for balance
    last_detection = None
    
    try:
        while True:
            frame = cap.read()
            current_time = time.time()
            
            # Inference throttling
            should_infer = (current_time - last_inference_time) >= inference_interval
            
            if should_infer:
                img = preprocess_frame(frame, target_size=(320, 320))
                
                # YOLO inference
                with torch.no_grad():
                    if device == 'cuda':
                        with torch.amp.autocast('cuda'):
                            results = model(img, size=320)
                    else:
                        results = model(img, size=320)
                
                detections = results.xyxy[0].cpu().numpy()
                person_boxes = [d for d in detections if int(d[5]) == 0 and d[4] > 0.4]  # Confidence threshold
                
                if person_boxes:
                    # Track largest person
                    largest = max(person_boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                    last_detection = largest
                    
                    # Map back to original frame coordinates
                    scale_x = frame.shape[1] / 320
                    scale_y = frame.shape[0] / 320
                    cx = int((largest[0] + largest[2]) / 2 * scale_x)
                    
                    servo_controller.update(cx, frame.shape[1])
                
                last_inference_time = current_time
            
            # Visualization
            if last_detection is not None:
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 320
                x1, y1, x2, y2, conf, cls = last_detection
                
                box_x1 = int(x1 * scale_x)
                box_y1 = int(y1 * scale_y)
                box_x2 = int(x2 * scale_x)
                box_y2 = int(y2 * scale_y)
                
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                cx = (box_x1 + box_x2) // 2
                cy = (box_y1 + box_y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{conf:.2f}", (box_x1, box_y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            fps = fps_counter.update()
            cv2.putText(frame, f"FPS: {fps:.1f} | Servo: {servo_controller.current_angle}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Single Person Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 User interrupted")
    finally:
        cv2.destroyAllWindows()

def track_fencers(target_fps=30):
    fps_counter = FPSCounter()
    last_inference_time = 0
    inference_interval = 1.0 / 15
    last_detections = None
    
    # Track the first two people detected and lock onto them
    tracked_fencers = None  # Will store reference positions of the two fencers
    fencer_lock_threshold = 100  # pixels - how far a person can move and still be considered the same
    
    def match_detection_to_tracked(detection, tracked_list):
        """Find which tracked fencer (if any) matches this detection"""
        det_cx = (detection[0] + detection[2]) / 2
        det_cy = (detection[1] + detection[3]) / 2
        
        best_match = None
        best_distance = float('inf')
        
        for i, tracked in enumerate(tracked_list):
            if tracked is None:
                continue
            tr_cx = (tracked[0] + tracked[2]) / 2
            tr_cy = (tracked[1] + tracked[3]) / 2
            
            distance = np.sqrt((det_cx - tr_cx)**2 + (det_cy - tr_cy)**2)
            if distance < best_distance and distance < fencer_lock_threshold:
                best_distance = distance
                best_match = i
        
        return best_match
    
    try:
        while True:
            frame = cap.read()
            current_time = time.time()
            
            should_infer = (current_time - last_inference_time) >= inference_interval
            
            if should_infer:
                img = preprocess_frame(frame, target_size=(320, 320))
                
                with torch.no_grad():
                    if device == 'cuda':
                        with torch.amp.autocast('cuda'):
                            results = model(img, size=320)
                    else:
                        results = model(img, size=320)
                
                detections = results.xyxy[0].cpu().numpy()
                person_boxes = [d for d in detections if int(d[5]) == 0 and d[4] > 0.35]
                
                # Initialize tracking with first two people detected
                if tracked_fencers is None and len(person_boxes) >= 2:
                    # Sort by x-coordinate and take leftmost and rightmost
                    person_boxes.sort(key=lambda x: (x[0]+x[2])/2)
                    tracked_fencers = [person_boxes[0], person_boxes[-1]]
                    last_detections = tracked_fencers.copy()
                    print("🎯 Locked onto two fencers!")
                
                # Update tracked fencers positions
                elif tracked_fencers is not None and len(person_boxes) > 0:
                    updated_fencers = [None, None]
                    used_detections = set()
                    
                    # Match current detections to tracked fencers
                    for detection in person_boxes:
                        match_idx = match_detection_to_tracked(detection, tracked_fencers)
                        if match_idx is not None and match_idx not in used_detections:
                            updated_fencers[match_idx] = detection
                            used_detections.add(match_idx)
                    
                    # Update tracked positions (keep old position if not found)
                    for i in range(2):
                        if updated_fencers[i] is not None:
                            tracked_fencers[i] = updated_fencers[i]
                    
                    # Only update servo if we have both fencers
                    if updated_fencers[0] is not None and updated_fencers[1] is not None:
                        last_detections = [tracked_fencers[0], tracked_fencers[1]]
                        
                        scale_x = frame.shape[1] / 320
                        cx_left = int((tracked_fencers[0][0] + tracked_fencers[0][2]) / 2 * scale_x)
                        cx_right = int((tracked_fencers[1][0] + tracked_fencers[1][2]) / 2 * scale_x)
                        mid_x = (cx_left + cx_right) // 2
                        
                        servo_controller.update(mid_x, frame.shape[1])
                
                last_inference_time = current_time
            
            # Visualization
            if last_detections is not None and len(last_detections) >= 2:
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 320
                
                centers = []
                for idx, box in enumerate(last_detections):
                    x1, y1, x2, y2, conf, cls = box
                    box_x1 = int(x1 * scale_x)
                    box_y1 = int(y1 * scale_y)
                    box_x2 = int(x2 * scale_x)
                    box_y2 = int(y2 * scale_y)
                    
                    # Different colors for each fencer
                    color = (0, 255, 0) if idx == 0 else (255, 0, 255)
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 2)
                    cx = (box_x1 + box_x2) // 2
                    centers.append(cx)
                    
                    # Label fencers
                    label = f"F{idx+1}: {conf:.2f}"
                    cv2.putText(frame, label, (box_x1, box_y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if len(centers) >= 2:
                    mid_x = sum(centers) // len(centers)
                    cv2.line(frame, (mid_x, 0), (mid_x, frame.shape[0]), (0, 0, 255), 2)
            
            # Display tracking status
            status = "🎯 LOCKED" if tracked_fencers is not None else "🔍 SEARCHING..."
            cv2.putText(frame, status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            fps = fps_counter.update()
            cv2.putText(frame, f"FPS: {fps:.1f} | Servo: {servo_controller.current_angle}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Fencers Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Press 'r' to reset tracking
                tracked_fencers = None
                print("🔄 Tracking reset - searching for new fencers...")
                
    except KeyboardInterrupt:
        print("\n🛑 User interrupted")
    finally:
        cv2.destroyAllWindows()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("OPTIMIZED TRACKING SYSTEM")
    print("="*50)
    mode = input("Select mode: [1] Single Person, [2] Fencers: ")
    
    if mode == "1":
        track_single_person(target_fps=30)
    elif mode == "2":
        track_fencers(target_fps=30)
    else:
        print("Invalid selection")
    
    cap.release()
    sock.close()
    print("✅ Clean exit")