###
# File: /DITracker/models/dnn.py
# Created Date: Tuesday, September 24th 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 24th September 2024 12:27:14 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import torch.nn as nn
import numpy as np
from .model_factory import register_model

@register_model('dnn')
class DNN(nn.Module):
    def __init__(self, input_dim, logger=None, hidden_layers=None):
        super(DNN, self).__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]  # 默认隐藏层

        input_dim = int(np.prod(input_dim))  # 展平输入维度
        layers = []
        for h in hidden_layers:
            layers.extend((nn.Linear(input_dim, h), nn.ReLU()))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
