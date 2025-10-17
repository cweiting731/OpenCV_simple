import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import os

class EdgeDetectionGroup:
    img = None
    sobel_x = None
    sobel_x_raw = None
    sobel_y = None
    sobel_y_raw = None
    def __init__(self, ui):
        self.ui = ui
        # 綁定按鈕事件
        self.ui.btn_three_load_image.clicked.connect(self.load_image)
        self.ui.btn_three_clear_image.clicked.connect(self.clear_image)
        self.ui.btn_sobel_x.clicked.connect(self.sobel_x_generator)
        self.ui.btn_sobel_y.clicked.connect(self.sobel_y_generator)
        self.ui.btn_combination_and_threshold.clicked.connect(self.combination_and_threshold)
        self.ui.btn_gradient_angle.clicked.connect(self.gradient_angle)
        self.ui.label_three_image.setText("No Image Loaded")
        self.ui.label_three_image.setStyleSheet("color: red;")
        self.ui.btn_sobel_x.setEnabled(False)
        self.ui.btn_sobel_y.setEnabled(False)
        self.ui.btn_combination_and_threshold.setEnabled(False)
        self.ui.btn_gradient_angle.setEnabled(False)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.img = cv2.imread(path)
            self.ui.label_three_image.setText(os.path.basename(path))
            self.ui.label_three_image.setStyleSheet("color: green;")
            self.ui.btn_sobel_x.setEnabled(True)
            self.ui.btn_sobel_y.setEnabled(True)
            self.ui.btn_combination_and_threshold.setEnabled(True)
            self.ui.btn_gradient_angle.setEnabled(True)

    def clear_image(self):
        self.img = None
        self.ui.label_three_image.setText("No Image Loaded")
        self.ui.label_three_image.setStyleSheet("color: red;")
        self.ui.btn_sobel_x.setEnabled(False)
        self.ui.btn_sobel_y.setEnabled(False)
        self.ui.btn_combination_and_threshold.setEnabled(False)
        self.ui.btn_gradient_angle.setEnabled(False)

    def filter2D(self, img, filter):
        h, w = img.shape
        result = np.zeros((h, w), dtype=np.float32)

        padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i + 3, j:j + 3] * filter)

        return result

    def sobel_x_generator(self):
        if self.sobel_x is not None:
            cv2.imshow("Sobel X", self.sobel_x)
            return
        if self.img is None:
            return
        # 1) Convert the RGB image into a grayscale image
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 2) Smooth grayscale image with Gaussian smoothing filter
        blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0)
        # 3) Apply Sobel edge detection to detect vertical edge by your own 3x3 Sobel x operator.
        sobel_x_filter = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])
        self.sobel_x_raw = self.filter2D(blurred, sobel_x_filter)
        self.sobel_x = np.clip(self.sobel_x_raw, 0, 255).astype(np.uint8)
        cv2.imshow("Sobel X", self.sobel_x)
        cv2.imshow

    def sobel_y_generator(self):
        if self.sobel_y is not None:
            cv2.imshow("Sobel Y", self.sobel_y)
            return
        if self.img is None:
            return
        
        # 1) Convert the RGB image into a grayscale image
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 2) Smooth grayscale image with Gaussian smoothing filter
        blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0)
        # 3) Apply Sobel edge detection to detect horizontal edge by your own 3x3 Sobel y operator.
        sobel_y_filter = np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]])
        self.sobel_y_raw = self.filter2D(blurred, sobel_y_filter)
        self.sobel_y = np.clip(self.sobel_y_raw, 0, 255).astype(np.uint8)
        cv2.imshow("Sobel Y", self.sobel_y)

    def combination_and_threshold(self):
        if self.img is None:
            return
        if self.sobel_x is None:
            self.sobel_x_generator()
        if self.sobel_y is None:
            self.sobel_y_generator()
        sobel_x = self.sobel_x.astype(np.float32)
        sobel_y = self.sobel_y.astype(np.float32)

        combination = np.sqrt(sobel_x**2 + sobel_y**2)
        combination = np.clip(combination, 0, 255).astype(np.uint8)
    
        _, th128 = cv2.threshold(combination, 128, 255, cv2.THRESH_BINARY)
        _, th28 = cv2.threshold(combination, 28, 255, cv2.THRESH_BINARY)

        cv2.imshow("Combination", combination)
        cv2.imshow("Threshold 128", th128)
        cv2.imshow("Threshold 28", th28)

    def gradient_angle(self):
        if self.img is None:
            return
        if self.sobel_x_raw is None:
            self.sobel_x_generator()
        if self.sobel_y_raw is None:
            self.sobel_y_generator()

        radian = np.arctan2(self.sobel_y_raw, self.sobel_x_raw)
        print(np.min(radian), np.max(radian))
        angle = np.rad2deg(radian)
        angle[angle < 0] += 360  # 將角度範圍調整到 [0, 360)

        mask_170_190 = ((angle >= 170) & (angle <= 190)).astype(np.uint8) * 255
        mask_260_280 = ((angle >= 260) & (angle <= 280)).astype(np.uint8) * 255

        output_170_190 = cv2.bitwise_and(self.img, self.img, mask=mask_170_190)
        output_260_280 = cv2.bitwise_and(self.img, self.img, mask=mask_260_280)

        cv2.imshow("mask 170-190", mask_170_190)
        cv2.imshow("mask 260-280", mask_260_280)
        cv2.imshow("Gradient Angle 170-190", output_170_190)
        cv2.imshow("Gradient Angle 260-280", output_260_280)