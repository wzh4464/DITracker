import torch.nn as nn
from .model_factory import register_model
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@register_model("cnn")
class CNN(nn.Module):
    def __init__(self, input_dim, logger=None, channels=[32, 64]):
        """
        Initialize CNN model for image classification tasks.

        Args:
            input_dim (tuple): 输入图像的维度 (in_channels, height, width)，如 (1, 28, 28)。
            channels (list): 每个卷积层的输出通道数量，默认为 [32, 64]。
        """
        super(CNN, self).__init__()
        in_channels, height, width = input_dim

        # 定义卷积层
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 计算全连接层输入维度
        fc_input_dim = channels[1] * (height // 4) * (width // 4)

        # 定义全连接层
        self.fc_layer = nn.Sequential(
            nn.Linear(fc_input_dim, 1),  # 输出为1个单元，用于二分类
            nn.Sigmoid(),  # Sigmoid 函数限制输出为 [0, 1]
        )

        # 初始化 logger
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            f"CNN initialized with input_dim={input_dim} and channels={channels}"
        )

    def forward(self, x):
        """
        Forward pass of CNN model.

        Args:
            x (torch.Tensor): 输入张量，通常为 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: 输出张量，二分类概率。
        """
        self.logger.debug(f"CNN forward input shape: {x.shape}")
        x = self.conv_layer(x)
        self.logger.debug(f"Shape after conv layers: {x.shape}")
        x = x.view(x.size(0), -1)  # 展平
        self.logger.debug(f"Shape after flattening: {x.shape}")
        x = self.fc_layer(x)
        self.logger.debug(f"Final output shape: {x.shape}")
        return x


@register_model("cnn_cifar")
class CNN_CIFAR(nn.Module):
    def __init__(self, input_dim, logger=None, m=[32, 32, 64, 64, 128, 128]):
        """
        Initialize the CNN for CIFAR dataset.

        Args:
            input_dim (tuple): 输入图像的维度 (height, width, channels)，例如 (32, 32, 3) 对于 CIFAR 数据集。
            logger (logging.Logger): 日志记录器，如果没有传递则使用默认的 logger。
            m (list): 每个卷积层的输出通道数量，默认为 [32, 32, 64, 64, 128, 128]。
        """
        super(CNN_CIFAR, self).__init__()
        height, width, in_channels = input_dim  # 输入图像的维度
        self.m = m

        # 定义卷积层
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=m[0], kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(m[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[0], out_channels=m[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=m[1], out_channels=m[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[2], out_channels=m[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=m[3], out_channels=m[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[4], out_channels=m[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 全连接层
        fc_input_dim = (height // 8) * (width // 8) * m[5]
        self.fc_layer = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # 输出层，用于二分类
            nn.Sigmoid(),  # 使用 Sigmoid 激活函数
        )

        # 初始化 logger
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            f"CNN_CIFAR initialized with input_dim={input_dim} and channels={m}"
        )

    def forward(self, x):
        """
        前向传播函数，将输入通过卷积层和全连接层得到输出。

        Args:
            x (torch.Tensor): 输入张量，通常为 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: 输出张量，二分类结果。
        """
        self.logger.debug(f"CNN_CIFAR forward input shape: {x.shape}")
        x = x.permute(
            0, 3, 1, 2
        )  # 转换输入的维度为 (batch_size, channels, height, width)
        self.logger.debug(f"Shape after permute: {x.shape}")

        x = self.conv_layer(x)  # 通过卷积层
        self.logger.debug(f"Shape after conv layers: {x.shape}")

        x = x.reshape(x.size(0), -1)  # 展平
        self.logger.debug(f"Shape after flattening: {x.shape}")

        x = self.fc_layer(x)  # 通过全连接层
        self.logger.debug(f"Final output shape: {x.shape}")

        return x
