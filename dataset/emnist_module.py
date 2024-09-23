###
# File: /DITracker/dataset/emnist_module.py
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

from emnist import extract_training_samples
import numpy as np
import sys
import os

# 动态添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.base_module import DataModule  # 现在可以直接导入 core.py 了


class EMNISTModule(DataModule):
    def load_data(self):
        """Load the EMNIST dataset for letters."""
        self.logger.info("Loading EMNIST Letters dataset")

        # 加载 EMNIST 数据集的 'letters' 子集
        x_train, y_train = extract_training_samples("letters")

        # 只保留类 'A' (标签 1) 和 'B' (标签 2)
        mask = (y_train == 1) | (y_train == 2)
        x_train = x_train[mask].astype(np.float32) / 255.0
        y_train = y_train[mask]

        # 将标签转换为二分类 (A=0, B=1)
        y_train = (y_train == 2).astype(np.int32)

        # 保留图片的二维结构并添加通道维度，形状为 (samples, 28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)

        self.logger.info(
            f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}"
        )
        return x_train, y_train
