import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from typing import List, Optional
import numpy as np


class TemporalGCNLayer(MessagePassing):
    """时空图卷积层，含残差连接"""

    def __init__(self, in_channels: int, out_channels: int, aggr: str = 'mean'):
        super().__init__()
        self.aggr = aggr
        # print(in_channels, out_channels)
        self.feature_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x: 节点特征 [batch_size, num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
        Returns:
            输出特征 [batch_size, num_nodes, out_channels]
        """
        # 空间聚合
        # print(self.aggr)
        aggregated = self.propagate(edge_index, x=x)

        # 残差连接与特征变换
        residual = aggregated + x  # [B, N, C_in]
        out = F.relu(self.feature_proj(residual))  # [B, N, C_out]
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class TemporalPredictor(nn.Module):
    """时序预测模块（LSTM核心）"""

    def __init__(self, input_dim: int, hidden_dim: int, pred_steps: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, pred_steps),
            nn.Unflatten(-1, (pred_steps, 1)) ) # 增加特征维度

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 输入序列 [batch_size*num_nodes, seq_len, input_dim]
        Returns:
            预测结果 [batch_size*num_nodes, pred_steps, 1]
        """
        lstm_out, _ = self.lstm(x)  # [*, seq_len, hidden_dim]
        last_step = lstm_out[:, -1, :]  # 取最终时间步 [*, hidden_dim]
        return self.decoder(last_step)  # [*, pred_steps, 1]


class TSGN(nn.Module):
    """时空图神经网络（完整模型）"""

    def __init__(self,
                 gcn_dims: List[int],
                 lstm_hidden: int,
                 pred_steps: int,
                 aggrs: Optional[List[str]] = None,
                 ):
                 # device: str = 'cuda'):
        super(TSGN, self).__init__()
        # self.device = device

        # 验证参数
        assert len(gcn_dims) >= 2, "GCN需要至少输入输出维度"
        self.gcn_layers = len(gcn_dims) - 1
        self.aggrs = aggrs if aggrs else ['mean'] * len(gcn_dims)
        self.hidden_dim = lstm_hidden
        self.pred_steps = pred_steps
        # 构建GCN模块
        self.gcn = nn.ModuleList([
            TemporalGCNLayer(in_channels=gcn_dims[i], out_channels=gcn_dims[i + 1], aggr=self.aggrs)
            for i in range(self.gcn_layers)
        ])

        # 构建时序模块
        self.temporal_predictor = TemporalPredictor(
            input_dim=gcn_dims[-1],
            hidden_dim=lstm_hidden,
            pred_steps=pred_steps
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x_seq: 输入序列 [batch_size, seq_len, num_nodes, 1]
            edge_index: 边索引 [2, num_edges]
        Returns:
            预测结果 [batch_size, pred_steps, num_nodes, 1]
        """
        batch_size, seq_len, num_nodes, _ = x_seq.shape

        # 空间特征提取 (并行化处理)
        spatial_features = []
        for t in range(seq_len):
            # 当前时刻特征 [B, N, 1] -> [B, N, C]
            x_t = x_seq[:, t, :, :]
            for gcn_layer in self.gcn:
                x_t = gcn_layer(x_t, edge_index)
            spatial_features.append(x_t.unsqueeze(1))  # [B, 1, N, C]

        # 组合时空特征 [B, T, N, C] -> [B, N, T, C]
        spatio_temporal = torch.cat(spatial_features, dim=1)
        spatio_temporal = spatio_temporal.permute(0, 2, 1, 3)  # [B, N, T, C]

        # 重塑为LSTM输入格式 [B*N, T, C]
        lstm_input = spatio_temporal.reshape(-1, seq_len, spatio_temporal.size(-1))

        # 时序预测 [B*N, pred_steps, 1]
        pred = self.temporal_predictor(lstm_input)

        # 恢复原始维度 [B, pred_steps, N, 1]
        pred = pred.reshape(batch_size, num_nodes, -1, 1)
        return pred.permute(0, 2, 1, 3)  # [B, pred_steps, N, 1]

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lstm_hidden", type=int, default=256)
        parser.add_argument("--pred_steps", type=int, default=7)
        return parser

    @property
    def hyperparameters(self):
        return {
            "gcn_layers": self.gcn_layers,
            "aggrs": self.aggrs,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.pred_steps,
        }


if __name__ == "__main__":
    x =  np.load('/home/jameszhang/桌面/SST/data/all_data/sst_mmap.npy',
                   mmap_mode='r', allow_pickle=True)[0:30,600:800, 400:600]
    mask = np.load('/home/jameszhang/桌面/SST/data/data_gradient/gradient_mask.npz')["mask"]
    print(x.shape, mask.shape)
    X = torch.from_numpy(x[:,mask]).float()
    # X = torch.randn(num_nodes, timesteps)
    print(X.shape)
    X = X.permute(1,0)
    # 生成edge_index
    # edge_index = generate_edge_index(
    #     X,
    #     threshold=0.9,
    #     top_k=10,
    #     device='cuda'
        # device='cpu',
    # )
    # print(edge_index.shape)
    # print(f"Generated edge_index shape: {edge_index.shape}")
    # print(f"Example edges:\n{edge_index[:, :5]}")
    # 参数配置
    # config = {
    #     'gcn_dims': [1, 32, 64],  # 输入特征维度1，两层GCN
    #     'lstm_hidden': 128,
    #     'pred_steps': 4,
    #     'aggrs': ['mean', 'max'],
    #     # 'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    #     'device': 'cpu'
    # }
    #
    # # 初始化模型
    # model = TSGN(**config).to(config['device'])
    # print(f"Model Architecture:\n{model}")
    #
    # # 测试输入
    # batch_size = 4
    # seq_len = 28
    # num_nodes = edge_index.shape[1]  # 测试用较小节点数
    # dummy_input = torch.randn(
    #     batch_size, seq_len, num_nodes, 1
    # ).to(config['device'])
    # # edge_index = torch.randint(0, num_nodes, (2, 500)).to(config['device'])

    # 前向传播
    # with torch.no_grad():
    #     output = model(dummy_input, edge_index)
    #     print(f"Input shape:  {tuple(dummy_input.shape)}")
    #     print(f"Output shape: {tuple(output.shape)}")
    #     assert output.shape == (batch_size, config['pred_steps'], num_nodes, 1)


