import cv2
import torch
import numpy as np
import socket
import time
import warnings
from collections import deque
from threading import Thread, Lock
import queue
import gc
gc.enable()  # Enable garbage collection

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
torch.cuda.empty_cache()
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.to(device)
model.eval()
model.conf = 0.35  # Lower confidence threshold for better detection
model.iou = 0.45   # Lower IoU threshold
model.classes = [0]  # Only detect people class
model.max_det = 10  # Limit maximum detections

# Optimize model
if device == 'cuda':
    model.half()  # FP16 for speed
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere
    torch.cuda.set_per_process_memory_fraction(0.8)  # Limit GPU memory usage
    
    # Try to enable TensorRT optimization
    try:
        print("Attempting TensorRT optimization...")
        model = torch.jit.script(model)
        print("✅ TensorRT optimization enabled")
    except Exception as e:
        print(f"ℹ️ TensorRT optimization not available: {e}")

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
#ESP32_IP = "172.20.10.6"
if ESP32_IP is None:
    print("❌ Could not resolve ESP32 hostname. Make sure your PC is on the same WiFi as ESP32.")
    exit()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
try:
    sock.connect((ESP32_IP, ESP32_PORT))
    print(f"✅ Connected to ESP32 at {ESP32_IP}:{ESP32_PORT}")
    
    # Initialize servo to center position (90°)
    time.sleep(0.1)  # Brief delay to ensure connection is stable
    sock.sendall("90\n".encode())
    print("🎯 Servo initialized to center position (90°)")
    time.sleep(0.5)  # Give servo time to reach position
    
except Exception as e:
    print("❌ Could not connect to ESP32:", e)
    exit()

# -----------------------------
# Camera setup with threading
# -----------------------------
class VideoCapture:
    """Threaded video capture with frame dropping for reduced latency"""
    def __init__(self, src=0, backend=cv2.CAP_MSMF):
        self.cap = cv2.VideoCapture(src, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG format
        self.frame_count = 0  # Track frames for potential dropping
        
        self.q = queue.Queue(maxsize=2)
        self.lock = Lock()
        self.stopped = False
        
        # Warm up
        for _ in range(5):
            self.cap.read()
        
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()
    
    def _reader(self):
        last_read_time = time.time()
        min_frame_time = 1.0 / 30
        
        while not self.stopped:
            current_time = time.time()
            elapsed = current_time - last_read_time
            
            if elapsed < min_frame_time:
                time.sleep(0.001)
                continue
            
            # Check if queue has space BEFORE reading
            if self.q.full():
                time.sleep(0.001)
                continue
                
            last_read_time = current_time
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Simpler queue management
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass
    
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
camera_indices = [1,2,3,4]
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
    print("❌ Could not open any camera. ")
    exit()

time.sleep(0.5)

# -----------------------------
# Servo PID with Kalman filter and adaptive speed
# -----------------------------
SERVO_INVERTED = True  # Set to True if servo rotates opposite direction

class ServoController:
    def __init__(self):
        self.current_angle = 90
        self.previous_error = 0
        self.integral = 0
        self.history = deque(maxlen=5)  # Much longer history for heavy smoothing
        self.angle_history = deque(maxlen=10)  # Track desired angles
        
        # Base PID gains (will be scaled based on proximity)
        self.k_p_base = 0.1  # Reduced from 0.06 for less aggressive response
        self.k_i = 0.0  # Disabled integral to prevent drift
        self.k_d_base = 0.8  # Increased from 0.2 for more damping
        self.deadzone_base = 10  # Increased from 40 to reduce jitter
        self.max_step_base = 4  # Reduced from 3 for smoother movement
        
        # Adaptive parameters
        self.k_p = self.k_p_base
        self.k_d = self.k_d_base
        self.deadzone = self.deadzone_base
        self.max_step = self.max_step_base
        self.min_angle_change = 2  # Minimum angle change to actually send command
        
        # Simple Kalman filter for position smoothing
        self.kalman_gain = 0.5
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
    
    def adapt_speed_to_proximity(self, box_height, frame_height):
        """Adjust movement speed based on detected person size (proximity indicator)"""
        # Normalized size: larger box = closer person = faster movement needed
        size_ratio = box_height / frame_height
        
        # Speed multiplier: reduced ranges to prevent overshooting
        if size_ratio > 0.5:  # Very close
            speed_multiplier = 2.0
        elif size_ratio > 0.35:  # Close
            speed_multiplier = 1.5
        elif size_ratio > 0.25:  # Medium distance
            speed_multiplier = 1.2
        elif size_ratio > 0.15:  # Far
            speed_multiplier = 1.0
        else:  # Very far
            speed_multiplier = 0.7
        
        # Apply multiplier to control parameters
        self.k_p = self.k_p_base * speed_multiplier
        self.k_d = self.k_d_base * speed_multiplier
        self.max_step = int(self.max_step_base * speed_multiplier)
        self.deadzone = max(25, int(self.deadzone_base / speed_multiplier))
        
        return speed_multiplier
    
    def update(self, target_x, frame_width, box_height=None, frame_height=480):
        # Adapt speed based on proximity if box height is provided
        speed_mult = 1.0
        if box_height is not None and frame_height is not None:
            speed_mult = self.adapt_speed_to_proximity(box_height, frame_height)
        
        # Apply Kalman filter
        filtered_x = self.kalman_update(target_x)
        
        # Moving average
        self.history.append(filtered_x)
        smoothed_target = int(np.mean(self.history))
        
        # PID control
        error = smoothed_target - frame_width // 2
        
        # Invert error if servo is wired backwards
        if SERVO_INVERTED:
            error = -error
        
        if abs(error) > self.deadzone:
            self.integral = np.clip(self.integral + error, -100, 100)
            derivative = error - self.previous_error
            
            delta = int(self.k_p * error + self.k_i * self.integral + self.k_d * derivative)
            delta = np.clip(delta, -self.max_step, self.max_step)
            
            target_angle = np.clip(self.current_angle + delta, 0, 180)
            # Smoothly interpolate toward the target
            new_angle = self.current_angle + 0.3 * (target_angle - self.current_angle)
            
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_command_time >= self.min_command_interval:
                if abs(new_angle - self.current_angle) > 0:
                    try:
                        sock.sendall(f"{int(new_angle)}\n".encode())
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

def preprocess_frame(frame, target_size=(224, 224)):
    """More efficient preprocessing"""
    # Skip frames if too large
    if frame.shape[0] > 640 or frame.shape[1] > 640:
        frame = cv2.resize(frame, (640, 480))
    
    # Use faster resize method
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

def track_single_person(target_fps=30):
    fps_counter = FPSCounter()
    last_inference_time = 0
    inference_interval = 1.0 / 30
    last_detection = None
    detection_timeout = 0.5
    last_detection_time = 0
    last_gc_time = time.time()
    gc_interval = 10.0
    
    try:
        while True:
            frame = cap.read()
            current_time = time.time()
            
            # Periodic garbage collection
            if current_time - last_gc_time > gc_interval:
                gc.collect()
                last_gc_time = current_time
            
            # Clear stale detection if too old
            if last_detection is not None and (current_time - last_detection_time) > detection_timeout:
                last_detection = None
            
            # Inference throttling
            should_infer = (current_time - last_inference_time) >= inference_interval
            
            if should_infer:
                img = preprocess_frame(frame, target_size=(224, 224))
                
                # Add warmup frames if no recent detection
                if time.time() - last_detection_time > 1.0:
                    with torch.no_grad():
                        model(img, size=224)
                        model(img, size=224)
                
                # YOLO inference
                with torch.no_grad():
                    if device == 'cuda':
                        with torch.amp.autocast('cuda'):
                            results = model(img, size=224)
                    else:
                        results = model(img, size=224)
                
                # Process detections more efficiently
                detections = results.xyxy[0]
                if device == 'cuda':
                    detections = detections.cpu()
                detections = detections.numpy()
                
                # Vectorized filtering for person class and confidence
                mask = (detections[:, 5] == 0) & (detections[:, 4] > 0.4)
                person_boxes = detections[mask]
                
                if person_boxes.shape[0] > 0:
                    # Track largest person
                    largest = max(person_boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                    last_detection = largest
                    last_detection_time = current_time
                    
                    # Map back to original frame coordinates
                    scale_x = frame.shape[1] / 320
                    scale_y = frame.shape[0] / 320
                    cx = int((largest[0] + largest[2]) / 2 * scale_x)
                    box_height = int((largest[3] - largest[1]) * scale_y)
                    
                    servo_controller.update(cx, frame.shape[1], box_height, frame.shape[0])
                else:
                    last_detection = None
                
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
            cv2.putText(frame, f"FPS: {fps:.1f} | Servo: {servo_controller.current_angle:.0f}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            #cv2.imshow("Single Person Tracking", frame)
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
    detection_timeout = 0.5
    last_detection_time = 0
    
    # Track the first two people detected and lock onto them
    tracked_fencers = None
    fencer_lock_threshold = 100
    
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
            
            # Clear stale detections if too old
            if last_detections is not None and (current_time - last_detection_time) > detection_timeout:
                last_detections = None
            
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
                    person_boxes.sort(key=lambda x: (x[0]+x[2])/2)
                    tracked_fencers = [person_boxes[0], person_boxes[-1]]
                    last_detections = tracked_fencers.copy()
                    print("🎯 Locked onto two fencers!")
                
                # Update tracked fencers positions
                elif tracked_fencers is not None and len(person_boxes) > 0:
                    updated_fencers = [None, None]
                    used_detections = set()
                    
                    for detection in person_boxes:
                        match_idx = match_detection_to_tracked(detection, tracked_fencers)
                        if match_idx is not None and match_idx not in used_detections:
                            updated_fencers[match_idx] = detection
                            used_detections.add(match_idx)
                    
                    for i in range(2):
                        if updated_fencers[i] is not None:
                            tracked_fencers[i] = updated_fencers[i]
                    
                    # Only update servo if we have both fencers
                    if updated_fencers[0] is not None and updated_fencers[1] is not None:
                        last_detections = [tracked_fencers[0], tracked_fencers[1]]
                        last_detection_time = current_time
                        
                        scale_x = frame.shape[1] / 320
                        scale_y = frame.shape[0] / 320
                        cx_left = int((tracked_fencers[0][0] + tracked_fencers[0][2]) / 2 * scale_x)
                        cx_right = int((tracked_fencers[1][0] + tracked_fencers[1][2]) / 2 * scale_x)
                        mid_x = (cx_left + cx_right) // 2
                        
                        height_left = int((tracked_fencers[0][3] - tracked_fencers[0][1]) * scale_y)
                        height_right = int((tracked_fencers[1][3] - tracked_fencers[1][1]) * scale_y)
                        avg_height = (height_left + height_right) // 2
                        
                        servo_controller.update(mid_x, frame.shape[1], avg_height, frame.shape[0])
                    else:
                        last_detections = None
                
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
                    
                    color = (0, 255, 0) if idx == 0 else (255, 0, 255)
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 2)
                    cx = (box_x1 + box_x2) // 2
                    centers.append(cx)
                    
                    label = f"F{idx+1}: {conf:.2f}"
                    cv2.putText(frame, label, (box_x1, box_y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if len(centers) >= 2:
                    mid_x = sum(centers) // len(centers)
                    cv2.line(frame, (mid_x, 0), (mid_x, frame.shape[0]), (0, 0, 255), 2)
            
            status = "🎯 LOCKED" if tracked_fencers is not None else "🔍 SEARCHING..."
            cv2.putText(frame, status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            fps = fps_counter.update()
            cv2.putText(frame, f"FPS: {fps:.1f} | Servo: {servo_controller.current_angle:.0f}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Fencers Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
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