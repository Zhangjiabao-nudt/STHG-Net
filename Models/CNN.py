from turtle import TurtleGraphicsError

import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch

# 核心网络架构
class SpatioTemporalCNN(nn.Module):
    def __init__(self, input_channels=1, output_timesteps=7):
        """
        Args:
            output_timesteps (int): 输出的时间步数量 t（如7）
        """
        super().__init__()
        self.output_timesteps = output_timesteps
        self.input_channels = input_channels
        # 时空特征编码器
        self.encoder = nn.Sequential(
            # 输入形状: (B, C=1, T, H, W)
            nn.Conv3d(input_channels, 16, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # 时间维度转换层
        self.time_proj = nn.Conv3d(
            in_channels=64,
            out_channels=output_timesteps,  # 直接投影到目标时间步数
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )

        # 空间特征精调层（保持维度）
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(output_timesteps, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, output_timesteps, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 输入形状调整: (B, T, H, W, 1) -> (B, 1, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)

        # 特征提取
        x = self.encoder(x)  # (B, 64, T, H, W)

        # 时间维度投影
        x = self.time_proj(x)  # (B, t, T, H, W)
        x = torch.mean(x, dim=2)  # (B, t, H, W)

        # 空间精调（保持时间维度为t）
        return self.spatial_refine(x)  # (B, t, H, W)

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--out", type=int, default=7)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self.input_channels, "hidden_dim": self.output_timesteps}


if __name__ == '__main__':
    model = SpatioTemporalCNN()
    print(model)
