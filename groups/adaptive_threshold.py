import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import os

class AdaptiveThresholdGroup:
    img = None
    def __init__(self, ui):
        self.ui = ui
        # 綁定按鈕事件
        self.ui.btn_five_load_image.clicked.connect(self.load_image)
        self.ui.btn_five_clear_image.clicked.connect(self.clear_image)
        self.ui.btn_global_threshold.clicked.connect(self.apply_global_threshold)
        self.ui.btn_local_threshold.clicked.connect(self.apply_local_threshold)
        self.ui.label_five_image.setText("No Image Loaded")
        self.ui.label_five_image.setStyleSheet("color: red;")
        self.ui.btn_global_threshold.setEnabled(False)
        self.ui.btn_local_threshold.setEnabled(False)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.ui.label_five_image.setText(os.path.basename(path))
            self.ui.label_five_image.setStyleSheet("color: green;")
            self.ui.btn_global_threshold.setEnabled(True)
            self.ui.btn_local_threshold.setEnabled(True)

    def clear_image(self):
        self.img = None
        self.ui.label_five_image.setText("No Image Loaded")
        self.ui.label_five_image.setStyleSheet("color: red;")
        self.ui.btn_global_threshold.setEnabled(False)
        self.ui.btn_local_threshold.setEnabled(False)

    def apply_global_threshold(self):
        if self.img is None:
            return
        if (self.img.ndim == 3):
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, thresh_img = cv2.threshold(self.img, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow('Global Threshold', thresh_img)

    def apply_local_threshold(self):
        if self.img is None:
            return
        if (self.img.ndim == 3):
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 19, -1)
        cv2.imshow('Local Threshold', thresh_img)