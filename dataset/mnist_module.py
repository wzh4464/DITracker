###
# File: /DITracker/dataset/mnist_module.py
# Created Date: Monday, September 23rd 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 24th September 2024 12:30:41 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import tensorflow as tf
import numpy as np
import sys
import os

# 动态添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.base_module import DataModule


class MnistModule(DataModule):
    def load_data(self):
        self.logger.info("Loading MNIST data")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # 合并训练和测试数据
        x = np.vstack([x_train, x_test]).astype(np.float32) / 255.0
        y = np.hstack([y_train, y_test])

        # 只保留数字 "1" 和 "7" 的样本
        mask = (y == 1) | (y == 7)
        x = x[mask]
        y = y[mask]

        # 将标签转换为二分类 (1=0, 7=1)
        y = (y == 7).astype(np.int32)

        # 为兼容 CNN，将 x 形状调整为 (samples, 28, 28, 1)
        x = x.reshape(-1, 28, 28, 1)

        self.logger.info(f"x shape: {x.shape}, y shape: {y.shape}")
        return x, y
