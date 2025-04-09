from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QRadioButton, QTableWidget, QTableWidgetItem
)

class HistoryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

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
