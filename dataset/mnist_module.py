###
# File: /DITracker/dataset/mnist_module.py
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

import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.base_module import DataModule


class MnistModule(DataModule):
    def load_data(self):
        self.logger.info("Loading MNIST data")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Stack the training and test data
        x = np.vstack([x_train, x_test]).astype(np.float32) / 255.0
        y = np.hstack([y_train, y_test])

        # Reshape x to have a channel dimension for CNN compatibility
        x = x.reshape(-1, 28, 28, 1)

        self.logger.info(f"x shape: {x.shape}, y shape: {y.shape}")
        return x, y
