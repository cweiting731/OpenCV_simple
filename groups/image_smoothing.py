import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import os

class ImageSmoothingGroup:
    img1 = None
    img2 = None
    def __init__(self, ui):
        self.ui = ui
        # 綁定按鈕事件
        self.ui.btn_two_load_image_1.clicked.connect(lambda: self.load_image(1))
        self.ui.btn_two_load_image_2.clicked.connect(lambda: self.load_image(2))
        self.ui.btn_two_clear_image_1.clicked.connect(lambda: self.clear_image(1))
        self.ui.btn_two_clear_image_2.clicked.connect(lambda: self.clear_image(2))
        self.ui.label_two_image_1.setText("No Image Loaded")
        self.ui.label_two_image_1.setStyleSheet("color: red;")
        self.ui.label_two_image_2.setText("No Image Loaded")
        self.ui.label_two_image_2.setStyleSheet("color: red;")
        self.ui.btn_gaussian_blur.clicked.connect(self.gaussian_blur)
        self.ui.btn_bilateral_filter.clicked.connect(self.bilateral_filter)
        self.ui.btn_median_filter.clicked.connect(self.median_blur)
        self.ui.btn_gaussian_blur.setEnabled(False)
        self.ui.btn_bilateral_filter.setEnabled(False)
        self.ui.btn_median_filter.setEnabled(False)


    def load_image(self, imgNum):
        path, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            if imgNum == 1:
                self.img1 = cv2.imread(path)
                self.ui.label_two_image_1.setText(os.path.basename(path))
                self.ui.label_two_image_1.setStyleSheet("color: green;")
                self.ui.btn_gaussian_blur.setEnabled(True)
                self.ui.btn_bilateral_filter.setEnabled(True)
            elif imgNum == 2:
                self.img2 = cv2.imread(path)
                self.ui.label_two_image_2.setText(os.path.basename(path))
                self.ui.label_two_image_2.setStyleSheet("color: green;")
                self.ui.btn_median_filter.setEnabled(True)

    def clear_image(self, imgNum):
        if imgNum == 1:
            self.img1 = None
            self.ui.label_two_image_1.setText("No Image Loaded")
            self.ui.label_two_image_1.setStyleSheet("color: red;")
            self.ui.btn_gaussian_blur.setEnabled(False)
            self.ui.btn_bilateral_filter.setEnabled(False)
        elif imgNum == 2:
            self.img2 = None
            self.ui.label_two_image_2.setText("No Image Loaded")
            self.ui.label_two_image_2.setStyleSheet("color: red;")
            self.ui.btn_median_filter.setEnabled(False)

    def gaussian_blur(self):
        if self.img1 is None:
            return
        def update(val):
            m = cv2.getTrackbarPos('m', 'Gaussian')
            kernel = 2 * m + 1
            blur = cv2.GaussianBlur(self.img1, (kernel, kernel), 0)
            cv2.imshow('Gaussian', blur)
        
        cv2.namedWindow('Gaussian')
        cv2.createTrackbar('m', 'Gaussian', 0, 10, update)
        update(1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bilateral_filter(self):
        if self.img1 is None:
            return
        def update(val):
            m = cv2.getTrackbarPos('m', 'Bilateral')
            d = 2 * m + 1
            sigmaColor = 90
            sigmaSpace = 90
            blur = cv2.bilateralFilter(self.img1, d, sigmaColor, sigmaSpace)
            cv2.imshow('Bilateral', blur)
        
        cv2.namedWindow('Bilateral')
        cv2.createTrackbar('m', 'Bilateral', 0, 10, update)
        update(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def median_blur(self):
        if self.img2 is None:
            return
        def update(val):
            m = cv2.getTrackbarPos('m', 'Median')
            kernel = 2 * m + 1
            blur = cv2.medianBlur(self.img2, kernel)
            cv2.imshow('Median', blur)

        cv2.namedWindow('Median')
        cv2.createTrackbar('m', 'Median', 0, 5, update)
        update(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
