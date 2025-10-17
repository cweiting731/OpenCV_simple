import sys
import os
import PyQt5
from PyQt5.QtWidgets import QApplication, QMainWindow
from hw1_ui import Ui_MainWindow

from groups.image_processing import ImageProcessingGroup
from groups.image_smoothing import ImageSmoothingGroup
from groups.edge_detection import EdgeDetectionGroup

# 這一段解決 "Could not find the Qt platform plugin 'windows'" 問題
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.path.dirname(PyQt5.__file__), "Qt", "plugins")

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.image_processing_group = ImageProcessingGroup(self)
        self.image_smoothing_group = ImageSmoothingGroup(self)
        self.edge_detection_group = EdgeDetectionGroup(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
