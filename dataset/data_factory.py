###
# File: /DITracker/dataset/data_factory.py
# Created Date: Monday, September 23rd 2024
# Author: Zihan
# -----
# Last Modified: Monday, 23rd September 2024 9:00:24 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

from .mnist_module import MnistModule
from .adult_module import AdultModule
from .cifar_module import CifarModule
from .emnist_module import EMNISTModule

class DataModuleFactory:
    @staticmethod
    def create(key: str, **kwargs):
        """Factory method to create appropriate DataModule based on key."""
        data_modules = {
            "mnist": MnistModule,
            "adult": AdultModule,
            "cifar": CifarModule,
            "emnist": EMNISTModule,
        }
        if key not in data_modules:
            raise ValueError(f"Unknown data module: {key}")
        return data_modules[key](**kwargs)

if __name__ == "__main__":
    # 单元测试代码
    mnist_data = DataModuleFactory.create("mnist", normalize=True, append_one=False)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_data.fetch(n_tr=50000, n_val=10000, n_test=10000)
    print(f"Factory MNIST Data Loaded: x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    
    emnist_data = DataModuleFactory.create("emnist", normalize=True, append_one=False)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = emnist_data.fetch(n_tr=40000, n_val=10000, n_test=10000)
    print(f"Factory EMNIST Data Loaded: x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
