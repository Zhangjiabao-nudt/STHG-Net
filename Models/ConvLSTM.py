import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # i, f, g, o gates
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, hidden_state):
        h,c = hidden_state
        combined = torch.cat([x, h], dim=1)  # concat along channel dim
        gates = self.conv(combined)

        # Split into input, forget, cell and output gates
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, layers=3):
        super(ConvLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = layers
        self.layers = nn.ModuleList(
            [ConvLSTMCell(input_channels, hidden_channels, kernel_size) for _ in range(layers)]
        )

    def forward(self, x, hidden_state):
        h, c = hidden_state
        if h is None:
            h = torch.zeros_like(x)
            c = torch.zeros_like(x)

        for i in range(self.num_layers):
            h, c = self.layers[i](x, (h, c))
        return h, c


class ConvLSTMPredictor(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=64, output_timesteps=7,
                 encoder_kernel_sizes=[5, 3], decoder_kernel_sizes=[3, 5]):
        super(ConvLSTMPredictor, self).__init__()
        self.output_timesteps = output_timesteps
        self.hidden_channels = hidden_channels
        # Encoder (spatial feature extractor)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels // 2,
                      kernel_size=encoder_kernel_sizes[0], padding=encoder_kernel_sizes[0] // 2),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels // 2, hidden_channels,
                      kernel_size=encoder_kernel_sizes[1], padding=encoder_kernel_sizes[1] // 2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # ConvLSTM core
        self.conv_lstm = ConvLSTM(hidden_channels, hidden_channels)

        # Decoder (spatial reconstructor)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2,
                      kernel_size=decoder_kernel_sizes[0], padding=decoder_kernel_sizes[0] // 2),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels // 2, 1,  # output 1 channel per timestep
                      kernel_size=decoder_kernel_sizes[1], padding=decoder_kernel_sizes[1] // 2)
        )

        # Output projection for multiple timesteps
        self.output_proj = nn.Conv2d(1, output_timesteps, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, T, H, W, C=1)
        Returns:
            output tensor of shape (B, output_timesteps, H, W)
        """
        B, T, H, W, C = x.shape
        assert C == 1, "Input channel must be 1"

        # Spatial encoding
        x = x.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        encoded = self.encoder(x)  # (B*T, hidden_channels, H, W)
        encoded = encoded.view(B, T, self.hidden_channels, H, W)

        # Temporal processing with ConvLSTM
        h, c = None, None
        for t in range(T):
            h, c = self.conv_lstm(encoded[:, t], (h, c))

        # Decode the last hidden state
        decoded = self.decoder(h)  # (B, 1, H, W)

        # Project to multiple output timesteps
        output = self.output_proj(decoded)  # (B, output_timesteps, H, W)

        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_channels", type=int, default=64)
        parser.add_argument("--output_timesteps", type=int, default=7)
        parser.add_argument("--encoder_kernel_sizes", type=int, default=[5, 3])
        parser.add_argument("--decoder_kernel_sizes", type=int, default=[3, 5])
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self.input_channels, "hidden_dim": self.output_timesteps}
# 验证代码

# python main.py --model_name CNNGRU --max_epochs 200 --learning_rate 0.001 --weight_decay 0 --batch_size 1 --loss mse --settings grid --gpus 1 --log_path test_ConvLSTM --pre_len 7 --hidden_channels 64

if __name__ == '__main__':
    # 参数设置
    B, T, H, W, C = 4, 10, 64, 64, 1
    output_t = 7

    # 初始化模型
    model = ConvLSTMPredictor(
        input_channels=C,
        hidden_channels=64,
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