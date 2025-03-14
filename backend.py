from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
import threading

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model YOLO Pose
model = YOLO("best_yolo8pose.pt")

# Variabel Global
prev_state = "stand"
squat_count = 0
state_changed = False
total_frames = 0
squat_frames = 0
is_running = False  # Untuk mengontrol jalannya deteksi

def process_webcam():
    global prev_state, squat_count, state_changed, total_frames, squat_frames, is_running
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Tidak dapat mengakses kamera!")
        return

    print("🎥 Kamera berhasil dibuka...")

    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Tidak bisa membaca frame dari kamera!")
            break

        total_frames += 1
        results = model(frame, imgsz=640)
        detected_class = None  

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)

                if conf > 0.5:
                    detected_class = model.names[cls_id]

        # Perhitungan squat
        if detected_class == "squat":
            if prev_state == "stand":
                state_changed = True
            prev_state = "squat"
            squat_frames += 1
        elif detected_class == "stand":
            if prev_state == "squat" and state_changed:
                squat_count += 1
                state_changed = False
            prev_state = "stand"

        # Hitung persentase squat
        squat_percentage = (squat_frames / total_frames) * 100 if total_frames > 0 else 0

        # Kirim data ke frontend via WebSocket
        socketio.emit("update_count", {
            "count": squat_count,
            "percentage": round(squat_percentage, 2)
        })

        # Tampilkan hasil di jendela
        cv2.putText(frame, f"Squats: {squat_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("YOLO Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@socketio.on("start_tracking")
def start_tracking():
    global is_running
    if not is_running:
        is_running = True
        thread = threading.Thread(target=process_webcam, daemon=True)
        thread.start()

@socketio.on("stop_tracking")
def stop_tracking():
    global is_running
    is_running = False

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
