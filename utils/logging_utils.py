###
# File: /DITracker/utils/logging_utils.py
# Created Date: Monday, September 23rd 2024
# Author: Zihan
# -----
# Last Modified: Monday, 23rd September 2024 9:04:36 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import logging
import os


def setup_logging(name: str, seed: int, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 文件日志输出
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}_{seed}.log"))
    file_handler.setLevel(logging.INFO)

    # 控制台日志输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化输出
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加 handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
