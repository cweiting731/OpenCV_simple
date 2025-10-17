import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import os

class TransformsGroup:
    img = None
    def __init__(self, ui):
        self.ui = ui
        # 綁定按鈕事件
        self.ui.btn_four_load_image.clicked.connect(self.load_image)
        self.ui.btn_four_clear_image.clicked.connect(self.clear_image)
        self.ui.btn_transforms.clicked.connect(self.transform_image)
        self.ui.label_four_image.setText("No Image Loaded")
        self.ui.label_four_image.setStyleSheet("color: red;")
        self.ui.btn_transforms.setEnabled(False)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.img = cv2.imread(path)
            self.ui.label_four_image.setText(os.path.basename(path))
            self.ui.label_four_image.setStyleSheet("color: green;")
            self.ui.btn_transforms.setEnabled(True)

    def clear_image(self):
        self.img = None
        self.ui.label_four_image.setText("No Image Loaded")
        self.ui.label_four_image.setStyleSheet("color: red;")
        self.ui.btn_transforms.setEnabled(False)

    def transform_image(self):
        if self.img is None:
            return
        h, w, _ = self.img.shape

        def get_float_value(widget, default=0.0):
            try:
                text = widget.toPlainText().strip()  # 取得多行框文字並去空白
                return float(text) if text else default
            except (ValueError, AttributeError):
                return default


        angle = get_float_value(self.ui.txt_rotation, 0.0)
        scale = get_float_value(self.ui.txt_scaling, 1.0)
        tx = get_float_value(self.ui.txt_tx, 0.0)
        ty = get_float_value(self.ui.txt_ty, 0.0)

        print(f"Angle: {angle}, Scale: {scale}, Tx: {tx}, Ty: {ty}")

        center = (240, 200)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        print("Transformation Matrix:\n", M)
        transformed = cv2.warpAffine(self.img, M, (w, h))

        cv2.imshow('Original Image', self.img)
        cv2.imshow('Transformed Image', transformed)