import torch
import torch.nn as nn
import numpy as np


class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 动态计算维度分配
        num_sin_dims = (d_model + 1) // 2
        div_term = torch.exp(
            torch.arange(0, num_sin_dims).float() *
            (-np.log(10000.0) / d_model)
        )
        # 安全赋值
        sin_dims = 2 * torch.arange(num_sin_dims)
        cos_dims = 2 * torch.arange(d_model // 2) + 1

        valid_sin_dims = sin_dims[sin_dims < d_model]
        valid_cos_dims = cos_dims[cos_dims < d_model]

        self.pe[:, valid_sin_dims] = torch.sin(position * div_term[:len(valid_sin_dims)])
        self.pe[:, valid_cos_dims] = torch.cos(position * div_term[:len(valid_cos_dims)])
        self.register_buffer('_pe', self.pe)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        pe = self.pe.to(x.device)
        return x + pe[:seq_len, :].unsqueeze(0)