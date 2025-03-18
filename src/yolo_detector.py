# src/yolo_detector.py
import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

class DetectionThread(QThread):
    # Signal untuk mengirim data update ke UI, berupa dictionary
    updateData = pyqtSignal(dict)
    
    def __init__(self, mode="squat", parent=None):
        super().__init__(parent)
        self.mode = mode  # "squat" atau "plank"
        self.running = False
        # Pastikan path model sesuai dengan struktur folder (di folder models)
        self.model = YOLO("models/best_yolo8pose.pt")
        self.threshold = 0.5
        
    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
        self.running = True
        
        # Variabel untuk mode Squat
        prev_state = "stand"
        state_changed = False
        squat_count = 0
        squat_start_time = time.time()
        
        # Variabel untuk mode Plank
        plank_start_time = time.time()
        last_time = time.time()
        plank_active_time = 0.0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            current_time = time.time()
            results = self.model(frame, imgsz=640)
            detected_class = None
            # Iterasi untuk mendapatkan kelas yang terdeteksi (dengan threshold)
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf > self.threshold:
                        cls_id = int(box.cls)
                        detected_class = self.model.names[cls_id]
                        break
                if detected_class is not None:
                    break
                    
            if self.mode == "squat":
                # Logika deteksi squat:
                if detected_class == "squat":
                    if prev_state == "stand":
                        state_changed = True
                    prev_state = "squat"
                elif detected_class == "stand":
                    if prev_state == "squat" and state_changed:
                        squat_count += 1
                        state_changed = False
                    prev_state = "stand"
                duration = current_time - squat_start_time
                data = {
                    "mode": "squat",
                    "squat_count": squat_count,
                    "squat_duration": int(duration)
                }
                self.updateData.emit(data)
                
            elif self.mode == "plank":
                total_time = current_time - plank_start_time
                delta = current_time - last_time
                # Jika deteksi menunjukkan "plank", anggap sebagai waktu aktif plank
                if detected_class == "plank":
                    plank_active_time += delta
                last_time = current_time
                data = {
                    "mode": "plank",
                    "plank_total_time": int(total_time),
                    "plank_active_time": int(plank_active_time)
                }
                self.updateData.emit(data)
            
            self.msleep(100)  # Istirahat 100ms agar CPU tidak overuse
            
        self.cap.release()
        
    def stop(self):
        self.running = False
        self.wait()
