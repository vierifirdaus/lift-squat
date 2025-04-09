# file: main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from counter_tab import CounterTab
from history_tab import HistoryTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Exercise Tracker with History")
        self.resize(500, 600)

        self.tabWidget = QTabWidget()
        self.counterTab = CounterTab()
        self.historyTab = HistoryTab()

        self.counterTab.sessionFinished.connect(self.historyTab.add_record)

        self.tabWidget.addTab(self.counterTab, "Counter")
        self.tabWidget.addTab(self.historyTab, "History")

        self.setCentralWidget(self.tabWidget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
