###
# File: /DITracker/models/model_factory.py
# Created Date: Tuesday, September 24th 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 24th September 2024 12:11:10 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging

# 模型注册字典
MODEL_REGISTRY = {}

def register_model(key):
    """用于注册模型的装饰器"""
    def decorator(cls):
        MODEL_REGISTRY[key] = cls
        return cls
    return decorator

def get_model(key, input_dim, logger=None, **kwargs):
    """根据模型key从注册表中实例化模型"""
    if logger is None:
        logger = logging.getLogger(__name__)

    if key not in MODEL_REGISTRY:
        logger.error(f"Model '{key}' 未在注册表中找到.")
        raise ValueError(f"Model '{key}' 未在注册表中找到.")
    
    logger.debug(f"创建模型: {key}，输入维度: {input_dim}")
    return MODEL_REGISTRY[key](input_dim, logger, **kwargs)

# 自动注册模型
from .logreg import LogReg
from .dnn import DNN
from .cnn import CNN, CNN_CIFAR
