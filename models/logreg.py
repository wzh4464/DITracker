###
# File: /DITracker/models/logreg.py
# Created Date: Tuesday, September 24th 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 24th September 2024 12:14:33 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import torch
import torch.nn as nn
import numpy as np
from .model_factory import register_model


@register_model("logreg")
class LogReg(nn.Module):
    def __init__(self, input_dim, logger=None):
        super(LogReg, self).__init__()
        if isinstance(input_dim, (tuple, list, torch.Size)):
            input_dim = int(np.prod(input_dim))  # 展平多维输入
        self.model = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
