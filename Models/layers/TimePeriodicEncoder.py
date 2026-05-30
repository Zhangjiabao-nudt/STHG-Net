import torch
import torch.nn as nn
import numpy as np


class TimePeriodicEncoder(nn.Module):
    def __init__(self, d_model: int, period_freqs: list = [7, 30, 365]):
        super().__init__()
        self.d_model = d_model
        self.period_freqs = period_freqs

        # 动态计算 div_term 的长度（兼容奇偶）
        self.num_sin_dims = (d_model + 1) // 2  # 奇数时sin多一维
        self.num_cos_dims = d_model // 2

        # 生成频率项（仅生成足够覆盖sin维度的项）
        div_term = torch.exp(
            torch.arange(0, self.num_sin_dims).float() *
            (-np.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, t_indices: torch.LongTensor):
        pe = torch.zeros(*t_indices.shape, self.d_model, device=t_indices.device)

        for freq in self.period_freqs:
            pos_in_cycle = t_indices % freq
            pos_in_cycle = pos_in_cycle.unsqueeze(-1).float()

            # 安全赋值：限制索引范围
            sin_dims = 2 * torch.arange(self.num_sin_dims)
            cos_dims = 2 * torch.arange(self.num_cos_dims) + 1

            # 过滤有效索引（防止d_model较小时的越界）
            valid_sin_dims = sin_dims[sin_dims < self.d_model]
            valid_cos_dims = cos_dims[cos_dims < self.d_model]

            pe[..., valid_sin_dims] += torch.sin(pos_in_cycle * self.div_term[:len(valid_sin_dims)])
            pe[..., valid_cos_dims] += torch.cos(pos_in_cycle * self.div_term[:len(valid_cos_dims)])

        return pe