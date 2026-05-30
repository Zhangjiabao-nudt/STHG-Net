import argparse
from pyexpat import features

import torch
import torch.nn as nn
import torch.nn.functional as F
from click.core import batch
from sympy import pretty
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class TemporalGatedConv(nn.Module):
    """ 时间门控卷积层（因果卷积 + GLU）"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,  # 输入通道数翻倍（GLU机制）
            out_channels=out_channels * 2,  # 输出通道数翻倍（GLU分割）
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,  # 保持时间维度不变
            padding_mode='replicate'
        )
        # 确保因果性：仅使用左侧的padding（需自定义实现）
        # 此处简化处理，假设kernel_size为奇数，padding为对称

    def forward(self, x):
        # x shape: (batch_size, num_nodes, features)
        batch_size, num_nodes, features = x.shape
        x = x.unsqueeze(0).permute(1,2,3,0)  # (B, num_nodes, features)
        x = x.view(batch_size*num_nodes, features, -1)
        x_conv = self.conv(x)  # (B*N, 2*out_channels, channels)
        out, gate = torch.chunk(x_conv, 2, dim=1)  # 分割为两部分
        out = torch.sigmoid(gate) * (out+x)  # GLU门控
        out = out.view(batch_size, num_nodes, features*1)  # (B*N, seq_len, out_channels)
        # print(out.shape, "TEM")
        return out

# class STConvBlock(nn.Module):
#     """ ST-Conv块：双GCN + 时间门控卷积 """
#
#     def __init__(self, in_feats, out_feats, adj_matrix, adj_matrix_c):
#         super().__init__()
#         self.gcn1 = GCNConv(in_feats, out_feats)  # 基于距离的图
#         self.gcn2 = GCNConv(in_feats, out_feats)  # 动态模式图
#         self.temp_conv = TemporalGatedConv(out_feats, out_feats)
#         self.adj_matrix = adj_matrix  # 静态邻接矩阵（距离图）
#         self.adj_matrix_c = adj_matrix_c  # 动态邻接矩阵（模式图）
#         self.learned_weight = nn.Parameter(torch.randn(out_feats))
#
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, num_nodes, features)
#         batch_size, seq_len, num_nodes, feats = x.shape
#         x = x.reshape(batch_size * seq_len, feats, num_nodes)
#         print( x.shape, self.adj_matrix.shape, "forward")
#         # 双GCN分支
#         out1 = F.relu(self.gcn1(x, self.adj_matrix))  # (B*T, N, out_feats)
#         print(out1.shape, x.shape, self.adj_matrix.shape, "forward")
#         out2 = F.relu(self.gcn2(x, self.adj_matrix_c))
#         fused = (out1 + out2) * self.learned_weight  # 加权融合
#
#         # 时间维度处理
#         fused = fused.reshape(batch_size, seq_len, num_nodes, -1)
#         fused = fused.permute(0, 2, 1, 3)  # (B, N, T, out_feats)
#         fused = fused.reshape(batch_size * num_nodes, seq_len, -1)
#
#         # 时间门控卷积
#         out_temp = self.temp_conv(fused)  # (B*N, T, out_feats)
#         out_temp = out_temp.reshape(batch_size, num_nodes, seq_len, -1)
#         out_temp = out_temp.permute(0, 2, 1, 3)  # (B, T, N, out_feats)
#         return out_temp

class ManualGCNConv(nn.Module):
    """手动实现的GCN层，包含对称归一化处理"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 1. 邻接矩阵归一化处理
        # adj_matrix = self.normalize_adj(adj_matrix)
        # self.register_buffer('norm_adj', adj_matrix)  # 固定参数，不参与训练

        # 2. 定义可训练参数
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    @staticmethod
    def normalize_adj(adj):
        """对称归一化邻接矩阵: D^(-1/2) (A + I) D^(-1/2)"""
        # 添加自环
        adj = adj + torch.eye(adj.size(0), device=adj.device)

        # 计算度矩阵
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        # 对称归一化
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return norm_adj

    def forward(self, x, adj_matrix):
        """
        输入x形状: (batch_size , num_nodes, in_features* seq_len)
        输出形状:   (batch_size , num_nodes, out_features* seq_len)
        """
        # 矩阵乘法: (N, N) @ (N, F_in) -> (N, F_in)
        B, N, F = x.size()
        x = x.permute(1,0,2).reshape(N, F* B)
        # with torch.no_grad():
            # support = torch.matmul(adj_matrix, x)  # [N, F_in]

        support = torch.sparse.mm(adj_matrix, x)
        support = support.unsqueeze(0)
        # 线性变换: (N, F_in) @ (F_in, F_out) -> (N, F_out)
        output = torch.matmul(support, self.weight) + self.bias
        return output


class STConvBlock(nn.Module):
    """修正后的ST-Conv块，使用手动GCN实现"""

    def __init__(self, in_feats, out_feats):
        super().__init__()
        # 初始化两个GCN层（静态图和动态图）
        self.gcn1 = ManualGCNConv(in_feats, out_feats)
        self.gcn2 = ManualGCNConv(in_feats, out_feats)

        self.temp_conv = TemporalGatedConv(out_feats, out_feats)
        self.learned_weight = nn.Parameter(torch.randn(out_feats))

    def forward(self, x, adj_matrix, adj_matrix_c):
        """输入x形状: (batch_size, seq_len, num_nodes, features*seq_len)"""
        batch_size, num_nodes, feats = x.shape

        # 双分支GCN
        out1 = F.relu(self.gcn1(x, adj_matrix))  # [batch, N, F_out]
        out2 = F.relu(self.gcn2(x, adj_matrix_c))

        # 特征融合
        fused = (out1 + out2) * self.learned_weight  # [batch*seq_len, N, F_out]
        # print(fused.shape, "fused")
        # 恢复时间维度
        fused = fused.view(batch_size, num_nodes, -1).contiguous()
        # 时间门控卷积
        out_temp = self.temp_conv(fused)  # [batch, N, F_out]
        return out_temp


class STCFE(nn.Module):
    """ 时空相关特征提取模块 """

    def __init__(self, in_feats, hidden_feats, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            STConvBlock(in_feats if i == 0 else hidden_feats, hidden_feats)
            for i in range(num_blocks)
        ])
        self.final_gcn = ManualGCNConv(hidden_feats, hidden_feats)

    def forward(self, x, adj_matrix, adj_matrix_c):
        # x shape: (B, T_r, N, F)
        batch_size, seq_len, num_nodes, feats = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, seq_len*feats).contiguous()
        for block in self.blocks:
            x = block(x, adj_matrix, adj_matrix_c)  # 逐层处理
        # 最终GCN
        B, N, F = x.shape
        x = x.reshape(B, N, F)
        x = self.final_gcn(x, adj_matrix)
        return x


class BC(nn.Module):
    """ 偏差校正模块 """

    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            ManualGCNConv(in_feats , hidden_feats),
            ManualGCNConv(hidden_feats, hidden_feats // 2),
            ManualGCNConv(hidden_feats//2, in_feats),
        ])
        self.final_gcn = ManualGCNConv(in_feats, hidden_feats)

    def forward(self, x_e, adj_matrix):
        # x_e: (B, T_e, N, F), x_r: (B, T_r, N, F)
        B, T_e, N, f = x_e.shape
        x_e = x_e.permute(0, 2, 1, 3).reshape(B, N, f * T_e).contiguous()
        # print(x_e.shape, "x_e")
        for i, gcn in enumerate(self.gcn_layers):
            x_record = F.relu(gcn(x_e, adj_matrix)) if i == 0 else F.relu(gcn(x_record, adj_matrix))
        diff = x_e - x_record  # 计算差异
        correction = self.final_gcn(diff, adj_matrix)
        # print(correction.shape, "correction")
        return correction


class PDM(nn.Module):
    """ 周期性依赖挖掘模块 """

    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.gcn = ManualGCNConv(in_feats, hidden_feats)
        self.temp_conv = TemporalGatedConv(hidden_feats, hidden_feats)
        self.final_gcn = ManualGCNConv(hidden_feats, hidden_feats)

    def forward(self, x_a, adj_matrix):
        # x_a: (B, T_a, N, F)
        B, T_a, N, f = x_a.shape
        x_a = x_a.permute(0, 2, 1, 3).reshape(B, N, f * T_a).contiguous()
        x = F.relu(self.gcn(x_a, adj_matrix))
        x = self.temp_conv(x)
        x = self.final_gcn(x, adj_matrix)
        # print(x.shape, "pdm")
        return x


class OVPGCN(nn.Module):
    def __init__(self, in_feats_st, in_feats_bc, in_feats_pdm, hidden_feats, pred_len, out_feats):
        super().__init__()
        # 模块定义
        self.stcfe = STCFE(in_feats_st, hidden_feats)
        self.bc = BC(in_feats_bc, hidden_feats)
        self.pdm = PDM(in_feats_pdm, hidden_feats)
        # 融合权重
        self.W_r = nn.Parameter(torch.randn(pred_len, hidden_feats))
        self.W_a = nn.Parameter(torch.randn(pred_len, hidden_feats))
        # 输出层
        self.fc_1 = nn.Linear(hidden_feats, out_feats)
        self.fc_2 = nn.Linear(hidden_feats, out_feats)
        self.pred_len = pred_len
        self.final_gcn = ManualGCNConv(hidden_feats, hidden_feats)
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats

    def forward(self, x_r, x_e, x_a, adj_matrix_r, adj_matrix_e, adj_matrix_a, adj_matrix_p):
        # 各模块处理
        batch_size, seq_len, num_nodes, feats = x_r.shape
        F_r = self.stcfe(x_r, adj_matrix_r, adj_matrix_p)  # (B, T_r, N, H)
        F_e = self.bc(x_e, adj_matrix_e)  # (B, T_r, N, H)
        F_a = self.pdm(x_a, adj_matrix_a)  # (B, T_a, N, H)

        # print(F_r.shape, F_e.shape, F_a.shape, "ovpgcn")
        # 中间融合（STCFE + BC）
        F_combined = F_r + F_e  # (B, T_r, N, H)
        F_combined = self.final_gcn(F_combined, adj_matrix_r)
        # print(F_combined.shape, "muse")
        # 多模块融合
        y = self.fc_1(F_combined)  + self.fc_2(F_a)
        y = y.permute(0,2,1).view(batch_size, self.pred_len, num_nodes, -1)
        # print(y.shape, "muse")
        return y

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--pred_steps", type=int, default=7)
        return parser

    @property
    def hyperparameters(self):
        return {
            'hidden_dim': self.hidden_feats,
            'output_dim': self.out_feats,
        }

# 测试用例
if __name__ == "__main__":
    # 参数设置
    B, T_r, N, C = 2, 6, 100, 1
    static_adj = torch.rand(N, N)
    dynamic_adj = torch.rand(N, N)
    pre_len = 3
    # 初始化模型
    model = OVPGCN(
        in_feats_st=C*T_r,
        in_feats_bc=C*T_r,
        in_feats_pdm=C*T_r*3,
        hidden_feats=64,
        pred_len=3,
        out_feats=C*pre_len
    )
    # adj_matrix = static_adj,
    # adj_matrix_c = dynamic_adj,
    # 测试前向传播
    recent = torch.randn(B, T_r, N, C)
    earlier = torch.randn(B, T_r, N, C)
    periodic = torch.randn(B, 3 * T_r, N, C)  # 假设T_a=3*T_r

    output = model(recent, earlier, periodic, static_adj, dynamic_adj, static_adj, dynamic_adj)
    print(f"输入形状: Recent {recent.shape}, Earlier {earlier.shape}, Periodic {periodic.shape}")
    print(f"输出形状: {output.shape} => (B, T_pred, N, C)")
    # 设置对应的adj、data