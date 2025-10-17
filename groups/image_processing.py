import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import os

class ImageProcessingGroup:
    img = None
    def __init__(self, ui):
        self.ui = ui
        # 綁定按鈕事件
        self.ui.btn_color_separation.clicked.connect(self.color_separation)
        self.ui.btn_color_transformation.clicked.connect(self.color_transformation)
        self.ui.btn_one_load_image.clicked.connect(self.load_image)
        self.ui.btn_color_extraction.clicked.connect(self.color_extraction)
        self.ui.btn_one_clear_image.clicked.connect(self.clear_image)
        self.ui.label_one_image.setText("No Image Loaded")
        self.ui.label_one_image.setStyleSheet("color: red;")
        self.ui.btn_color_separation.setEnabled(False)
        self.ui.btn_color_transformation.setEnabled(False)
        self.ui.btn_color_extraction.setEnabled(False)
        

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.img = cv2.imread(path)
            self.ui.label_one_image.setText(os.path.basename(path))
            self.ui.label_one_image.setStyleSheet("color: green;")
            self.ui.btn_color_separation.setEnabled(True)
            self.ui.btn_color_transformation.setEnabled(True)
            self.ui.btn_color_extraction.setEnabled(True)

    def clear_image(self):
        self.img = None
        self.ui.label_one_image.setText("No Image Loaded")
        self.ui.label_one_image.setStyleSheet("color: red;")
        self.ui.btn_color_separation.setEnabled(False)
        self.ui.btn_color_transformation.setEnabled(False)
        self.ui.btn_color_extraction.setEnabled(False)

    def color_separation(self):
        if self.img is None:
            return
        b, g, r = cv2.split(self.img)
        zeros = np.zeros_like(b)
        b_img = cv2.merge([b, zeros, zeros])
        g_img = cv2.merge([zeros, g, zeros])
        r_img = cv2.merge([zeros, zeros, r])
        cv2.imshow("Blue", b_img)
        cv2.imshow("Green", g_img)
        cv2.imshow("Red", r_img)

    def color_transformation(self):
        if self.img is None:
            return
        cv_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(self.img)
        avg_gray = ((b/3 + g/3 + r/3)).astype(np.uint8)
        cv2.imshow("cv_gray", cv_gray)
        cv2.imshow("avg_gray", avg_gray)

    def color_extraction(self):
        if self.img is None:
            return

        # 1) Transform “rgb.jpg” from BGR format to HSV format
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # 2) Yellow-Green mask I1
        lower = np.array([18, 0, 25])     # HSV 下限
        upper = np.array([85, 255, 255])  # HSV 上限
        mask = cv2.inRange(hsv, lower, upper)

        # 3) Invert the mask I1
        mask_inv = cv2.bitwise_not(mask)

        # 4) Remove Yellow and Green color in the image to generate I2
        extracted = cv2.bitwise_and(self.img, self.img, mask=mask_inv)

        cv2.imshow("Original", self.img)
        cv2.imshow("Yellow-Green Mask", mask)
        cv2.imshow("Inverted Mask", mask_inv)
        cv2.imshow("Image without yellow-green", extracted)