###
# File: /DITracker/models/test_models.py
# Created Date: Tuesday, September 24th 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 24th September 2024 12:27:22 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model_factory import get_model


def test_model(model_key, input_dim):
    """
    测试指定模型的创建和前向传播。
    """
    try:
        print(f"Testing model: {model_key}")
        model = get_model(model_key, input_dim)

        # 随机生成一批测试输入数据
        if model_key.startswith("cnn"):
            # CNN 模型需要 4D 输入 (batch_size, channels, height, width)
            x = torch.randn(10, *input_dim)
        else:
            # 非 CNN 模型需要 2D 输入 (batch_size, input_dim)
            x = torch.randn(10, input_dim[0])

        # 执行前向传播
        output = model(x)

        # 输出结果的维度，确保模型的输出符合预期
        print(f"{model_key} output shape: {output.shape}")

        print(f"{model_key} Test Passed!")
        return True
    except Exception as e:
        print(f"{model_key} Test Failed! Error: {str(e)}")
        return False


if __name__ == "__main__":
    test_results = [("logreg", test_model("logreg", (28 * 28,)))]

    # 测试 DNN 模型
    test_results.append(("dnn", test_model("dnn", (28 * 28,))))

    # 测试 CNN 模型
    test_results.append(("cnn", test_model("cnn", (1, 28, 28))))

    # 测试 CNN_CIFAR 模型
    test_results.append(("cnn_cifar", test_model("cnn_cifar", (32, 32, 3))))

    # 汇总结果
    print("\n=== Test Summary ===")
    for model_name, result in test_results:
        status = "Passed" if result else "Failed"
        print(f"{model_name}: {status}")
