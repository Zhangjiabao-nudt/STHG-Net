import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# class CNNGRUPredictor(nn.Module):
#     def __init__(self, input_channels=1, output_timesteps=7,
#                  cnn_hidden=64, gru_hidden=256):
#         super().__init__()
#         self.output_timesteps = output_timesteps
#
#         # 空间编码器 (保持空间分辨率)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_channels, 16, kernel_size=5, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#
#             nn.Conv2d(16, cnn_hidden, kernel_size=5, padding=1),
#             nn.BatchNorm2d(cnn_hidden),
#             nn.ReLU()
#         )
#
#         # 延迟初始化的GRU参数
#         self.gru_hidden = gru_hidden
#         self.gru = None
#         self.fc = None
#
#         # 空间解码器
#         self.decoder = nn.Sequential(
#             nn.Conv2d(cnn_hidden, 16, kernel_size=5, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#
#             nn.Conv2d(16, 1, kernel_size=3, padding=1)
#         )
#
#     def forward(self, x):
#         B, T, H, W, C = x.shape
#
#         # 空间编码
#         x = x.view(B * T, C, H, W)
#         encoded = self.encoder(x)  # (B*T, cnn_hidden, H, W)
#
#         # 动态初始化GRU
#         if self.gru is None:
#             self._initialize_gru(encoded)
#
#         # 时空特征处理
#         gru_input = encoded.view(B, T, -1)  # (B, T, cnn_hidden*H*W)
#         _, hidden = self.gru(gru_input)  # 使用最后一个隐藏状态
#
#         # 生成预测序列
#         pred_features = self.fc(hidden).view(
#             B, self.output_timesteps, -1, H, W)  # (B, t, cnn_hidden, H, W)
#
#         # 空间解码
#         outputs = []
#         for t in range(self.output_timesteps):
#             dec = self.decoder(pred_features[:, t])  # (B, 1, H, W)
#             outputs.append(dec.squeeze(1))
#
#         return torch.stack(outputs, dim=1)  # (B, t, H, W)
#
#     def _initialize_gru(self, encoded):
#         cnn_hidden = encoded.size(1)
#         H = encoded.size(2)
#         W = encoded.size(3)
#
#         self.gru = nn.GRU(
#             input_size=cnn_hidden * H * W,
#             hidden_size=self.gru_hidden,
#             batch_first=True
#         ).to(encoded.device)
#
#         self.fc = nn.Linear(
#             self.gru_hidden,
#             self.output_timesteps * cnn_hidden * H * W
#         ).to(encoded.device)
#
#     @staticmethod
#     def add_model_specific_arguments(parent_parser):
#         parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
#         parser.add_argument("--out", type=int, default=7)
#         return parser
#
#     @property
#     def hyperparameters(self):
#         return {"input_dim": self.input_channels, "hidden_dim": self.output_timesteps}

# python main.py --model_name CNNGRU --max_epochs 200 --learning_rate 0.001 --weight_decay 0 --batch_size 1 --loss mse --settings grid --gpus 1 --log_path test_CNNGRU --pre_len 7
# 验证代码

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGRUPredictor(nn.Module):
    def __init__(self, input_channels=1, output_timesteps=7,
                 cnn_hidden=32, gru_hidden=128):
        super().__init__()
        self.output_timesteps = output_timesteps
        self.cnn_hidden = cnn_hidden

        # 空间编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),  # 保持尺寸
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, cnn_hidden, kernel_size=5, padding=2),
            nn.BatchNorm2d(cnn_hidden),
            nn.ReLU()
        )

        # 延迟初始化参数
        self.gru_hidden = gru_hidden
        self.gru = None
        self.fc = None
        self.initialized_shape = None  # 记录初始化时的空间尺寸

        # 空间解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(cnn_hidden, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """专业化的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, T, H, W, C = x.shape

        # 空间编码
        x = x.view(B * T, C, H, W)
        encoded = self.encoder(x)  # (B*T, C_hidden, H, W)

        # 动态初始化GRU（首次运行时执行）
        if self.gru is None:
            self._initialize_gru(encoded)
            self.initialized_shape = (H, W)
        else:
            # 确保空间尺寸一致
            assert (H, W) == self.initialized_shape, \
                f"输入尺寸{(H, W)}与初始化尺寸{self.initialized_shape}不匹配"

        # 时空特征处理
        _, C_hidden, H_enc, W_enc = encoded.shape
        gru_input = encoded.view(B, T, -1)  # (B, T, C_hidden*H*W)

        # GRU处理
        _, hidden = self.gru(gru_input)  # hidden: (num_layers, B, gru_hidden)
        hidden = hidden[-1]  # 取最后一层 hidden: (B, gru_hidden)

        # 生成预测特征
        pred_features = self.fc(hidden)
        pred_features = pred_features.view(
            B, self.output_timesteps, C_hidden, H_enc, W_enc
        )  # (B, T_out, C_hidden, H, W)

        # 空间解码
        outputs = []
        for t in range(self.output_timesteps):
            dec_out = self.decoder(pred_features[:, t])  # (B, 1, H, W)
            outputs.append(dec_out.squeeze(1))  # (B, H, W)

        return torch.stack(outputs, dim=1)  # (B, T_out, H, W)

    def _initialize_gru(self, sample_tensor):
        """根据样本特征动态初始化GRU"""
        _, C_hidden, H, W = sample_tensor.shape
        gru_input_size = C_hidden * H * W

        # 初始化GRU
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self.gru_hidden,
            batch_first=True,
            num_layers=2,  # 使用双层GRU增强时序建模
            dropout=0.2
        ).to(sample_tensor.device)

        # 初始化全连接层
        self.fc = nn.Linear(
            self.gru_hidden,
            self.output_timesteps * C_hidden * H * W
        ).to(sample_tensor.device)

        # GRU参数初始化
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # 遗忘门偏置初始化
                param.data[self.gru_hidden:2 * self.gru_hidden].fill_(1)

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--cnn_hidden", type=int, default=32)
        parser.add_argument("--gru_hidden", type=int, default=128)
        parser.add_argument("--out", type=int, default=7)
        return parser

    @property
    def hyperparameters(self):
        return {
            "input_channels": self.input_channels,
            "output_timesteps": self.output_timesteps,
            "cnn_hidden": self.cnn_hidden,
            "gru_hidden": self.gru_hidden
        }
if __name__ == '__main__':
    # 参数设置
    B, T, H, W, C = 4, 10, 64, 64, 1
    output_t = 7

    # 初始化模型
    model = CNNGRUPredictor(
        input_channels=C,
        output_timesteps=output_t
    )
    print("模型结构:")
    print(model)

    # 创建测试输入
    dummy_input = torch.randn(B, T, H, W, C)
    print("\n输入形状:", dummy_input.shape)

    # 前向传播
    output = model(dummy_input)
    print("输出形状:", output.shape)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 梯度检查
    dummy_input.requires_grad = True
    loss = output.mean()
    loss.backward()
    print("\n梯度检查:")
    print("输入梯度:", dummy_input.grad is not None)
    print("模型参数梯度:", all(p.grad is not None for p in model.parameters() if p.requires_grad))