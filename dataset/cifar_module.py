###
# File: /DITracker/dataset/cifar_module.py
# Created Date: Monday, September 23rd 2024
# Author: Zihan
# -----
# Last Modified: Monday, 23rd September 2024 9:55:53 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

# DITracker/dataset/cifar_module.py
import tensorflow as tf
import numpy as np
import sys
import os

# 动态添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.base_module import DataModule  # 导入 core 模块

# DITracker/dataset/cifar_module.py
import tensorflow as tf
import numpy as np
import os
from dataset.base_module import DataModule


class CifarModule(DataModule):
    def __init__(self, cifar_version=10, **kwargs):
        super().__init__(**kwargs)
        self.cifar_version = cifar_version

    def load_data(self):
        self.logger.info(f"Loading CIFAR-{self.cifar_version} data")
        if self.cifar_version == 10:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        elif self.cifar_version == 100:
            (x_train, y_train), (x_test, y_test) = (
                tf.keras.datasets.cifar100.load_data()
            )

        # Flatten the labels from (50000, 1) and (10000, 1) to (50000,) and (10000,)
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        x = np.vstack([x_train, x_test]).astype(np.float32) / 255.0
        y = np.hstack([y_train, y_test])

        # Select only class 0 and class 1 for binary classification
        mask = (y == 0) | (y == 1)
        x, y = x[mask], y[mask]
        y = (y == 1).astype(int)  # Convert to binary classification

        return x, y
