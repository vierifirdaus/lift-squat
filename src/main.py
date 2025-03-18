import sys
import time
import random
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QGroupBox, QLineEdit,
    QTabWidget, QTableWidget, QTableWidgetItem
)

##########################################
# QThread untuk YOLO Deteksi dan Kamera
##########################################
class DetectionThread(QThread):
    # Sinyal untuk mengirim data counter (dictionary) dan frame kamera (QImage)
    updateData = pyqtSignal(dict)
    updateFrame = pyqtSignal(QImage)

    def __init__(self, mode="squat", parent=None):
        super().__init__(parent)
        self.mode = mode  # "squat" atau "plank"
        self.running = False
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
            # Lakukan deteksi dengan YOLO dan dapatkan frame yang telah dianotasi
            results = self.model(frame, imgsz=640)
            annotated_frame = results[0].plot()
            detected_class = None

            # Iterasi untuk mendapatkan kelas dengan confidence tinggi
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf > self.threshold:
                        cls_id = int(box.cls)
                        detected_class = self.model.names[cls_id]
                        break
                if detected_class is not None:
                    break

            # Logika perhitungan berdasarkan mode
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

            # Tambahkan overlay label pada frame hasil deteksi (annotated_frame)
            box_x, box_y, box_w, box_h = 20, 20, 200, 80
            cv2.rectangle(annotated_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), -1)
            if self.mode == "squat":
                label_text = f"Squats: {squat_count}"
            else:
                label_text = f"Plank: {data.get('plank_active_time', 0)}/{data.get('plank_total_time', 0)}"
            cv2.putText(annotated_frame, label_text, (box_x + 10, box_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

            # Konversi frame yang telah dianotasi (dengan overlay) ke QImage
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

##########################################
# Tab Counter: Input Nama, Mode, Kamera, Data
##########################################
class CounterTab(QWidget):
    # Sinyal untuk mengirim record sesi ke tab History
    sessionFinished = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "squat"
        self.initUI()
        self.detection_thread = None
        self.latest_data = {}
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Input Nama
        name_layout = QHBoxLayout()
        name_label = QLabel("Nama:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        
        # Mode selection
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
        
        # Label untuk menampilkan feed kamera (ukuran besar di tengah)
        self.cameraLabel = QLabel("Camera Feed")
        self.cameraLabel.setFixedSize(640, 480)
        self.cameraLabel.setAlignment(Qt.AlignCenter)
        self.cameraLabel.setStyleSheet("border: 2px solid black;")
        camera_layout = QHBoxLayout()
        camera_layout.addStretch()
        camera_layout.addWidget(self.cameraLabel)
        camera_layout.addStretch()
        
        # Tombol Start dan Stop
        self.startButton = QPushButton("Start")
        self.startButton.clicked.connect(self.start_tracking)
        self.stopButton = QPushButton("Stop")
        self.stopButton.clicked.connect(self.stop_tracking)
        self.stopButton.setEnabled(False)
        
        # Label untuk menampilkan informasi counter/timer
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
        # Setelah sesi selesai, simpan record ke History
        name = self.name_input.text().strip() or "Unknown"
        session_data = {"name": name, "mode": self.mode}
        session_data.update(self.latest_data)
        self.sessionFinished.emit(session_data)
        
    def update_info(self, data):
        self.latest_data = data
        if data["mode"] == "squat":
            text = (f"Mode: Squat\nSquat Count: {data['squat_count']}\n"
                    f"Duration: {data['squat_duration']} sec")
        elif data["mode"] == "plank":
            text = (f"Mode: Plank\nPlank Active Time: {data['plank_active_time']} sec\n"
                    f"Total Time: {data['plank_total_time']} sec")
        self.infoLabel.setText(text)
        
    def update_camera(self, qImg):
        pixmap = QPixmap.fromImage(qImg)
        self.cameraLabel.setPixmap(pixmap.scaled(self.cameraLabel.size(), Qt.KeepAspectRatio))

##########################################
# Tab History: Menampilkan record sesi
##########################################
class HistoryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []  # Daftar record sesi
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Radio buttons untuk filter history
        self.historyGroupBox = QGroupBox("Pilih History")
        self.squatHistoryRadio = QRadioButton("Squat History")
        self.plankHistoryRadio = QRadioButton("Plank History")
        self.squatHistoryRadio.setChecked(True)
        self.squatHistoryRadio.toggled.connect(self.update_table)
        self.plankHistoryRadio.toggled.connect(self.update_table)
        history_mode_layout = QHBoxLayout()
        history_mode_layout.addWidget(self.squatHistoryRadio)
        history_mode_layout.addWidget(self.plankHistoryRadio)
        self.historyGroupBox.setLayout(history_mode_layout)
        
        # Tabel untuk menampilkan history
        self.table = QTableWidget()
        layout.addWidget(self.historyGroupBox)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.update_table()
        
    def add_record(self, record):
        self.history.append(record)
        self.update_table()
        
    def update_table(self):
        filter_mode = "squat" if self.squatHistoryRadio.isChecked() else "plank"
        filtered = [r for r in self.history if r["mode"] == filter_mode]
        if filter_mode == "squat":
            headers = ["Name", "Squat Count", "Squat Duration (sec)"]
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(headers)
            self.table.setRowCount(len(filtered))
            for i, record in enumerate(filtered):
                self.table.setItem(i, 0, QTableWidgetItem(record.get("name", "")))
                self.table.setItem(i, 1, QTableWidgetItem(str(record.get("squat_count", 0))))
                self.table.setItem(i, 2, QTableWidgetItem(str(record.get("squat_duration", 0))))
        else:
            headers = ["Name", "Plank Active Time (sec)", "Plank Total Time (sec)"]
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(headers)
            self.table.setRowCount(len(filtered))
            for i, record in enumerate(filtered):
                self.table.setItem(i, 0, QTableWidgetItem(record.get("name", "")))
                self.table.setItem(i, 1, QTableWidgetItem(str(record.get("plank_active_time", 0))))
                self.table.setItem(i, 2, QTableWidgetItem(str(record.get("plank_total_time", 0))))

##########################################
# Main Window dengan TabWidget
##########################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exercise Tracker with History")
        self.resize(500, 600)
        
        self.tabWidget = QTabWidget()
        self.counterTab = CounterTab()
        self.historyTab = HistoryTab()
        
        # Kirim record sesi dari CounterTab ke HistoryTab
        self.counterTab.sessionFinished.connect(self.historyTab.add_record)
        
        self.tabWidget.addTab(self.counterTab, "Counter")
        self.tabWidget.addTab(self.historyTab, "History")
        
        self.setCentralWidget(self.tabWidget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
