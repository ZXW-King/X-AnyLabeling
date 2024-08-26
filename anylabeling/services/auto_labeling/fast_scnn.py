
import logging
import os

import cv2
import math
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
import onnxruntime as ort
from .engines import OnnxBaseModel
from .types import AutoLabelingResult


class FastSCNN(Model):

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_run"
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "rotation": QCoreApplication.translate("Model", "Rotation"),
            "point": QCoreApplication.translate("Model", "Point"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        self.classes = self.config.get("classes", [])
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {self.config['type']} model.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)

    def img_crop(self,image, target_size):
        H, W, _ = image.shape
        diff_h = H - target_size[0]
        cropped_img = image[diff_h:, :]
        return cropped_img

    def preprocess(self, image):
        image = self.img_crop(image, (256, 640))
        self.img_height, self.img_width = image.shape[:2]
        # 将图像转换为 float32 类型并归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0

        # 定义均值和标准差
        mean = [0.485, 0.456, 0.406]  # 123.675  116.28  103.53
        std = [0.229, 0.224, 0.225]  # 58.395  57.12  57.375

        # 归一化
        image -= mean
        image /= std
        img = image.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype("float32")
        return img

    def get_sam_results(self, approx_contours, label=None):
        # Contours to shapes
        shapes = []
        if self.output_mode == "polygon":
            height, width = approx_contours.shape[:2]
            # Create shape
            shape = Shape(flags={})
            for x in range(height):
                for y in range(width):
                    if approx_contours[x, y] == 0:
                        shape.add_point(QtCore.QPointF(y, 224 + x))
            shape.shape_type = "polygon"
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = "AUTOLABEL_OBJECT" if label is None else label[0]
            shape.selected = False
            shapes.append(shape)
        return shapes if label is None else shapes[0]


    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """
        if image is None:
            return []

        try:
            cv_image = qt_img_to_rgb_cv_img(image, filename)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []
        model_abs_path = self.get_model_abs_path(
            self.config, "model_path"
        )
        if not model_abs_path or not os.path.isfile(
                model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of Segment Anything.",
                )
            )
        self.onnx_session = ort.InferenceSession(model_abs_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        input_img = self.preprocess(cv_image)
        input_feed = {"input":input_img}
        outs = self.onnx_session.run(['output'], input_feed=input_feed)
        shapes = []
        results = self.get_sam_results(outs[0],self.classes)
        shapes.append(results)
        result = AutoLabelingResult(shapes, replace=True)
        return result
