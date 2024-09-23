###
# File: /DITracker/dataset/test_datasets.py
# Created Date: Monday, September 23rd 2024
# Author: Zihan
# -----
# Last Modified: Monday, 23rd September 2024 9:54:46 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import toml
from mnist_module import MnistModule
from emnist_module import EMNISTModule
from cifar_module import CifarModule
from adult_module import AdultModule

# 读取配置文件
config = toml.load("dataset/config.toml")


def test_data_module(module, module_name):
    print(f"Testing module: {module_name}")

    # 获取配置
    n_tr = config[module_name]["n_tr"]
    n_val = config[module_name]["n_val"]
    n_test = config[module_name]["n_test"]

    # 预期形状
    if "input_shape" in config[module_name]:
        expected_shape = tuple(config[module_name]["input_shape"])
    else:
        expected_shape = (config[module_name]["num_features"],)

    # 加载数据
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = module.fetch(
        n_tr=n_tr, n_val=n_val, n_test=n_test
    )

    # 形状检查
    assert x_train.shape == (
        n_tr,
        *expected_shape,
    ), f"Train shape mismatch: {x_train.shape}"
    assert x_val.shape == (
        n_val,
        *expected_shape,
    ), f"Validation shape mismatch: {x_val.shape}"
    assert x_test.shape == (
        n_test,
        *expected_shape,
    ), f"Test shape mismatch: {x_test.shape}"

    assert y_train.shape == (n_tr,), f"Train labels shape mismatch: {y_train.shape}"
    assert y_val.shape == (n_val,), f"Validation labels shape mismatch: {y_val.shape}"
    assert y_test.shape == (n_test,), f"Test labels shape mismatch: {y_test.shape}"

    print(f"{module_name} Test Passed!")
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}\n")


if __name__ == "__main__":
    # 测试所有模块
    test_data_module(MnistModule(normalize=True, append_one=False), "MnistModule")
    test_data_module(EMNISTModule(normalize=True, append_one=False), "EMNISTModule")
    test_data_module(
        CifarModule(cifar_version=10, normalize=True, append_one=False), "CifarModule"
    )
    test_data_module(AdultModule(normalize=True, append_one=False), "AdultModule")
