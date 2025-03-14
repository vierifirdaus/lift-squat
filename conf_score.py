from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLO Pose
model = YOLO("model/best_yolo8pose.pt")

# Gunakan kamera laptop sebagai input
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

# Dapatkan informasi video dari kamera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  

# Buat output video
output_path = "output_yolo.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

prev_state = "stand"  # Mulai dari posisi berdiri
squat_count = 0
state_changed = False  # Untuk mendeteksi perubahan posisi

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Tidak bisa membaca frame dari kamera!")
        break  

    frame_count += 1
    results = model(frame, imgsz=640)
    annotated_frame = results[0].plot()  

    detected_class = None 

    for result in results:
        for box in result.boxes:  # Iterasi setiap bounding box
            cls_id = int(box.cls)  # ID kelas yang terdeteksi
            conf = float(box.conf)  # Confidence score

            if conf > 0.5:  # Gunakan threshold confidence untuk menghindari false detection
                detected_class = model.names[cls_id]  # Simpan kelas yang terdeteksi

    # Jika squat terdeteksi
    if detected_class == "squat":
        if prev_state == "stand":  # Transisi dari stand ke squat
            state_changed = True  # Tanda awal squat
        prev_state = "squat"
    
    # Jika kembali ke stand dari squat
    elif detected_class == "stand":
        if prev_state == "squat" and state_changed:
            squat_count += 1  # Hitung squat
            state_changed = False  # Reset perubahan status setelah hitung
        prev_state = "stand"

    # Tambahkan kotak jumlah squat pada frame
    box_x, box_y, box_w, box_h = 20, 20, 200, 80  # Posisi dan ukuran kotak
    cv2.rectangle(annotated_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), -1)  # Kotak hijau
    cv2.putText(annotated_frame, f"Squats: {squat_count}", (box_x + 10, box_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)  # Teks hitam

    # Tampilkan frame dengan anotasi secara real-time
    cv2.imshow('YOLO Pose Detection', annotated_frame)

    # Simpan frame yang telah diberi anotasi ke video output
    out.write(annotated_frame)

    # Tekan 'q' untuk keluar dari tampilan video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tampilkan hasil
print("\n========== Video Analysis Report YOLO 8 Pose ==========")
print(f"Total Frames Processed: {frame_count}")
print(f"Total Squats Detected: {squat_count}")

# Tutup video
cap.release()
out.release()
cv2.destroyAllWindows()
