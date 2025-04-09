import time
import cv2
import os, sys
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QGroupBox, QLineEdit
)


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class DetectionThread(QThread):
    updateData = pyqtSignal(dict)
    updateFrame = pyqtSignal(QImage)

    def __init__(self, mode="squat", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.running = False
        model_path = resource_path(os.path.join("models", "best_yolo8pose.pt"))
        self.model = YOLO(model_path)
        self.threshold = 0.5

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
        self.running = True

        prev_state = "stand"
        state_changed = False
        squat_count = 0
        squat_start_time = time.time()

        plank_start_time = time.time()
        last_time = time.time()
        plank_active_time = 0.0

        no_detection_start = None

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time()
            results = self.model(frame, imgsz=640)
            annotated_frame = results[0].plot()
            detected_class = None

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf > self.threshold:
                        cls_id = int(box.cls)
                        detected_class = self.model.names[cls_id]
                        break
                if detected_class is not None:
                    break

            if detected_class is None:
                if no_detection_start is None:
                    no_detection_start = current_time
                elif current_time - no_detection_start > 3:
                    data = {"mode": self.mode, "warning": "Pose tidak terdeteksi selama 3 detik!"}
                    self.updateData.emit(data)
                    if self.mode == "plank":
                        self.running = False
                        break
            else:
                no_detection_start = None

            if self.mode == "squat":
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
            elif self.mode == "plank":
                total_time = current_time - plank_start_time
                delta = current_time - last_time
                if detected_class == "plank":
                    plank_active_time += delta
                last_time = current_time
                data = {
                    "mode": "plank",
                    "plank_total_time": int(total_time),
                    "plank_active_time": int(plank_active_time)
                }

            self.updateData.emit(data)

            # box_x, box_y, box_w, box_h = 20, 20, 200, 80
            # cv2.rectangle(annotated_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), -1)
            # if self.mode == "squat":
            #     label_text = f"Squats: {squat_count}"
            # else:
            #     label_text = f"Plank: {data.get('plank_active_time', 0)}/{data.get('plank_total_time', 0)}"
            # cv2.putText(annotated_frame, label_text, (box_x + 10, box_y + 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_frame.shape
            bytesPerLine = channels * width
            qImg = QImage(rgb_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.updateFrame.emit(qImg)
            self.msleep(100)

        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

class CounterTab(QWidget):
    sessionFinished = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "squat"
        self.initUI()
        self.detection_thread = None
        self.latest_data = {}

    def initUI(self):
        layout = QVBoxLayout()

        name_layout = QHBoxLayout()
        name_label = QLabel("Nama:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)

        self.modeGroupBox = QGroupBox("Pilih Mode")
        self.squatRadio = QRadioButton("Squat")
        self.plankRadio = QRadioButton("Plank")
        self.squatRadio.setChecked(True)
        self.squatRadio.toggled.connect(self.change_mode)
        self.plankRadio.toggled.connect(self.change_mode)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.squatRadio)
        mode_layout.addWidget(self.plankRadio)
        self.modeGroupBox.setLayout(mode_layout)

        self.cameraLabel = QLabel("Camera Feed")
        self.cameraLabel.setFixedSize(640, 480)
        self.cameraLabel.setAlignment(Qt.AlignCenter)
        self.cameraLabel.setStyleSheet("border: 2px solid black;")
        camera_layout = QHBoxLayout()
        camera_layout.addStretch()
        camera_layout.addWidget(self.cameraLabel)
        camera_layout.addStretch()

        self.startButton = QPushButton("Start")
        self.startButton.clicked.connect(self.start_tracking)
        self.stopButton = QPushButton("Stop")
        self.stopButton.clicked.connect(self.stop_tracking)
        self.stopButton.setEnabled(False)

        self.infoLabel = QLabel("Mode: Squat\nSquat Count: 0\nDuration: 0 sec")
        self.infoLabel.setAlignment(Qt.AlignCenter)

        layout.addLayout(name_layout)
        layout.addWidget(self.modeGroupBox)
        layout.addLayout(camera_layout)
        layout.addWidget(self.startButton)
        layout.addWidget(self.stopButton)
        layout.addWidget(self.infoLabel)
        self.setLayout(layout)

    def change_mode(self):
        if self.squatRadio.isChecked():
            self.mode = "squat"
        elif self.plankRadio.isChecked():
            self.mode = "plank"
        self.infoLabel.setText(f"Mode: {self.mode.capitalize()}\nCounters reset.")

    def start_tracking(self):
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.detection_thread = DetectionThread(mode=self.mode)
        self.detection_thread.updateData.connect(self.update_info)
        self.detection_thread.updateFrame.connect(self.update_camera)
        self.detection_thread.start()

    def stop_tracking(self):
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        name = self.name_input.text().strip() or "Unknown"
        session_data = {"name": name, "mode": self.mode}
        session_data.update(self.latest_data)
        self.sessionFinished.emit(session_data)

    def update_info(self, data):
        self.latest_data = data
        if "warning" in data:
            self.infoLabel.setText(f"\u26a0\ufe0f WARNING: {data['warning']}")
            if self.detection_thread and not self.detection_thread.running:
                self.stop_tracking()
            return

        if data["mode"] == "squat":
            text = (f"Mode: Squat\nSquat Count: {data['squat_count']}\n"
                    f"Duration: {data['squat_duration']} sec")
        elif data["mode"] == "plank":
            text = (f"Mode: Plank\nPlank Active Time: {data['plank_active_time']} sec\n"
                    f"Total Time: {data['plank_total_time']} sec")
        self.infoLabel.setText(text)

        if data.get("mode") == "plank" and self.detection_thread is not None and not self.detection_thread.running:
            self.stop_tracking()

    def update_camera(self, qImg):
        pixmap = QPixmap.fromImage(qImg)
        self.cameraLabel.setPixmap(pixmap.scaled(self.cameraLabel.size(), Qt.KeepAspectRatio))
