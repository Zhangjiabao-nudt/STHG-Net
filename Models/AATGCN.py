import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from numpy.random import randn
from triton.language import dtype


class TemporalDifferencing(nn.Module):
    """时间差分处理模块（维度保持 BTNF）"""

    def forward(self, x):
        # x: [BT, N, F]
        x_diff = x[1:] - x[:-1]
        x_diff = torch.cat([x[:1], x_diff], dim=0)  # 保持维度
        return x_diff.unsqueeze(0).permute(0, 2, 1)


# class AAGCBlock(nn.Module):
#     """改进的注意力自适应图卷积模块"""
#
#     def __init__(self, in_feats, out_feats, num_heads=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = out_feats // num_heads
#
#         # 特征投影
#         self.fc = nn.Linear(in_feats, out_feats)
#
#         # 注意力计算
#         self.attn_fc = nn.Sequential(
#             nn.Linear(2 * self.head_dim, 1),
#             nn.LeakyReLU(0.2)
#         )
#
#         # 正则化
#         self.norm = nn.LayerNorm(out_feats)
#         self.dropout = nn.Dropout(0.1)
#
#     # def edge_attention(self, edges):
#     #     """边注意力计算（自动广播BT维度）"""
#     #     src_feat = edges.src['h']
#     #     dst_feat = edges.dst['h']
#     #     attn = self.attn_fc(torch.cat([src_feat, dst_feat], dim=-1))
#     #     print(attn.squeeze(-1).shape)
#     #     return {'e': attn.squeeze(-1)}  # [E, BT, H]
#
#     def edge_attention(self, edges):
#         # 输入特征维度：[E, 2*H]
#         # 注意力计算公式：a_ij = exp((u_i^T v_j) / sqrt(d))
#         u = edges.src['h']  # [E, H]
#         v = edges.dst['h']  # [E, H]
#         # alpha = torch.exp(torch.sum(u * v, dim=1, keepdim=True) / (u.shape[1] ** 0.5))  # [E, 1]
#         # print(alpha.squeeze(1).shape, "edge attention")
#         dot_product = torch.sum(u * v, dim=1, keepdim=True)  # [E, 1]
#         scaled_dot = dot_product / (self.head_dim ** 0.5)    # 缩放点积
#         alpha = torch.exp(scaled_dot - scaled_dot.max())     # 减去最大值防止溢出
#         return {'e': alpha.squeeze(1)}  # 输出维度必须为[E, 1]
#
#     def forward(self, x, g, training):
#         """
#         Args:
#             x: [BT, N, F]
#         Returns:
#             out: [BT, N, out_feats]
#         """
#         BT, N, _ = x.shape
#         # 特征投影
#         h = self.fc(x).view(BT, N, self.num_heads, -1)  # [BT, N, H, D]
#         # print(h.shape, "AAGCBlock")
#
#         with g.local_scope():
#             # 设置节点特征
#             g.ndata['h'] = h.permute(1, 0, 2, 3) # [N, BT, H, D]
#
#             # 计算边注意力
#             g.apply_edges(self.edge_attention)  # 输出[E, BT, H]
#
#             # 注意力权重归一化
#             g.edata['a'] = torch.sigmoid(g.edata['e'])
#
#             # 消息传递
#             g.update_all(fn.u_mul_e('h', 'a', 'm'),
#                          fn.sum('m', 'h_out'))
#
#             # 聚合结果
#             out = g.ndata['h_out'].permute(1, 0, 2, 3)  # [BT, N, H, D]
#             out = out.reshape(BT, N, -1)  # [BT, N, out_feats]
#             # print(out.shape, "AAGCBlock")
#             # 残差连接
#             return self.norm(out + self.fc(x))


class AAGCBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, beta=2 / 3, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(in_dim, hidden_dim)  # 特征变换 [F*T -> H]
        self.b = nn.Parameter(torch.randn(2 * hidden_dim))  # 注意力参数 [2H]
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.H = hidden_dim

    def hard_concrete(self, log_alpha, log_u, training=True):
        """生成硬混凝土分布掩码 (支持广播)"""
        # print(log_alpha.shape)
        if training:
            # u = torch.rand_like(log_alpha)
            # start = time.time()
            g = torch.sigmoid((log_u  + log_alpha) / self.beta)
            # print(time.time() - start)
            # start = time.time()
            g = g * (self.zeta - self.gamma) + self.gamma
            # print(time.time() - start)
        else:
            g = torch.sigmoid(log_alpha / self.beta) * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(g, 0, 1)

    def forward(self, x, adj, log_u, training=True):
        """
        Input:
            x:  [B, N, F*T]
            adj: [N, N]
        Output:
            out: [B, T, N, Hidden_dim]
        """
        B, N, F = x.shape
        # Step 1: 特征变换
        h = self.W(x)                             # [B, N, H]
        # print(1)
        # Step 2: 分解注意力系数计算（避免显式拼接）
        b1 = self.b[:self.hidden_dim]             # 前 H 维
        b2 = self.b[self.hidden_dim:]             # 后 H 维
        h_i_proj = torch.matmul(h, b1)            # [B, N]
        h_j_proj = torch.matmul(h, b2)            # [B, N]
        log_alpha = h_i_proj.unsqueeze(2) + h_j_proj.unsqueeze(1)  # [B, N, N]
        # print(2)
        # Step 3: 生成掩码并剪枝邻接矩阵
        # start = time.time()
        Z = self.hard_concrete(log_alpha, log_u, training)  # [B, N, N]
        # print(time.time() - start)
        # adj_expanded = adj.unsqueeze(0).expand(B, N, N) # [B, N, N]
        # adj_pruned = adj_expanded * Z              # 应用掩码剪枝
        # start = time.time()
        adj_pruned = adj.unsqueeze(0) * Z  # 利用广播机制合并前两行
        # print(time.time() - start)
        # start = time.time()
        rows = torch.arange(B, device=adj.device)[:, None, None]
        cols = torch.arange(N, device=adj.device)[None, :, None]
        adj_pruned[rows, cols, cols] = 1.0  # 张量索引设置对角线
        # print(time.time() - start)
        # adj_pruned[:, range(N), range(N)] = 1.0    # 保留自连接
        # print(3)
        # Step 4: 归一化注意力权重
        degrees = adj_pruned.sum(dim=2, keepdim=True)  # [B, N, 1]
        a = adj_pruned / (degrees + 1e-8)          # [B, N, N]
        # print(4)
        # Step 5: 特征聚合
        h_out = torch.bmm(a, h)                    # [B, N, H]
        h_out = h_out.reshape(B, N, self.hidden_dim)  # [B, T, N, H]
        return h_out


class TemporalConv(nn.Module):
    """时间卷积模块（处理合并的BT维度）"""

    def __init__(self, in_feats, out_feats, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_feats,
            out_feats,
            kernel_size=3,
            dilation=dilation,
            padding=dilation * (3 - 1) // 2
        )
        self.norm = nn.LayerNorm(out_feats)

    def forward(self, x):
        # x: [BT, N, C]
        x = x.permute(1, 2, 0)  # [N, C, BT]
        x = self.conv(x)[..., :-2]  # 因果卷积裁剪
        return self.norm(x.permute(2, 0, 1))  # [BT-2, N, C]


class TGCNBlock(nn.Module):
    """时间图卷积块（包含3个AAGC层）"""

    def __init__(self, in_feats, out_feats, dilation=1):
        super().__init__()
        self.aagc_layers = nn.ModuleList([
            AAGCBlock(in_feats if i == 0 else out_feats, out_feats)
            for i in range(3)
        ])
        # self.tconv = TemporalConv(out_feats, out_feats, dilation)
        self.res_conv = nn.Conv1d(in_feats, out_feats, 1) if in_feats != out_feats else None

    # def forward(self, x, g, training=True):
    def forward(self, x, g, log_u, training=True):
        # x: [BT, N, C]
        # residual = x
        for aagc in self.aagc_layers:
            y = aagc(x, g, log_u, training)
        # x = self.tconv(x)
        if self.res_conv is not None:
            x = self.res_conv(x.permute(1, 2, 0)).permute(2, 0, 1)
        # print(x.shape, residual.shape, "TGCNBlock")
        return F.relu(x + y)


class AA_TGCN(nn.Module):
    """完整的时空图卷积网络"""

    def __init__(self, input_steps=30, output_steps=7,
                 hidden_dim=64,  features_dim=1):
        super().__init__()
        # 时间差分处理
        self.diff = TemporalDifferencing()
        self.output_steps = output_steps
        self.features_dim = features_dim
        # 主干网络
        self.tgcn1 = TGCNBlock(input_steps*1, hidden_dim, dilation=1)
        self.tgcn2 = TGCNBlock(input_steps*1, hidden_dim, dilation=1)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_steps)
        )
    def forward(self, x, adj, log_u, training=True):
    # def forward(self, x, adj, training=True):
        """
        Args:
            x: [B, T, N, F] -> 输入时自动转换为[B, N, F*T]
        Returns:
            out: [B, t, N, F]
        """
        # 维度转换
        B, T, N, F = x.shape
        x_diff = self.diff(x.reshape(B*T, -1))

        x = x.permute(0, 2, 1, 3).view(B, N, -1)  # [B, N, F*T]
        # print(x.shape, "AA_TGCN")
        # 原始数据路径
        h_orig = self.tgcn1(x, adj, log_u, training)

        # 差分数据路径
        # print(x_diff.shape, "AA_TGCN")
        h_diff = self.tgcn2(x_diff, adj, log_u, training)

        # 特征融合
        h_fusion = h_orig + h_diff
        # 输出预测
        out = self.fc(h_fusion)  # [B, N, t]
        # print(out.shape, "AA_TGCN")
        return out.view(B, self.output_steps, N, -1)  # [B, t, N, F]


    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=64)
        # parser.add_argument('--seq_len', type=int, default=2)
        parser.add_argument("--pred_steps", type=int, default=7)
        parser.add_argument("--features_dim", type=int, default=1)
        return parser

    @property
    def hyperparameters(self):
        return {
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_steps,
        }

# --------------------------
# 工程化测试模块
# --------------------------
# def test_model():
#     # 构建测试图
#     g = dgl.graph(([0, 1], [1, 0]))  # 简单双向图
#
#     # 模型参数
#     B, T, N, F = 8, 12, 162, 1
#     t = 6
#
#     # 初始化模型
#     model = AA_TGCN(N, input_steps=T, output_steps=t)
#
#     # 测试输入
#     x = torch.randn(B, T, N, F)
#
#     # 前向传播
#     out = model(g, x)
#     print(f"输入维度: {x.shape} -> 输出维度: {out.shape}")
#     assert out.shape == (B, t, N, F), "维度验证失败！"

def create_complete_bipartite_graph(N):
    # 向量化生成所有可能边
    src = torch.arange(N).repeat(N)          # [0,0,0,1,1,1,...]
    dst = torch.arange(N).repeat_interleave(N) # [0,1,2,0,1,2,...]
    mask = src != dst                         # 排除自环边
    src, dst = src[mask], dst[mask]           # [0,0,1,1,2,2] 和 [1,2,0,2,0,1]
    return dgl.graph((src, dst))

if __name__ == "__main__":
    # 创建测试图
    g = dgl.graph(([0, 1], [1, 0]))  # 2个节点

    # 创建测试数据
    B, T, N, C = 8, 12, 2, 1
    h = torch.randn(T, N, 8, 1)  # [12, 2, 8, 1]

    # 执行修正代码
    h_flattened = h.permute(1, 0, 2, 3).reshape(N, -1)  # [2, 12*8*1] = [2, 96]
    g.ndata['h'] = h_flattened

    # 验证结果
    print("节点数:", g.num_nodes())  # 2
    print("特征形状:", g.ndata['h'].shape)  # torch.Size([2, 96])
    print("特征维度:", g.ndata['h'].dim())  # 2
    # a = torch.randn((1, 29021, 29021))
    # log_u = torch.log(a)-torch.log(1-a)
    # torch.save(log_u, "/home/jameszhang/log.pth")
    # test_model()
    # 构建测试图
    log_u = torch.load("/home/jameszhang/log.pth")
    # 模型参数
    B, T, N, C = 1, 30, 29021, 1
    t = 7
    # g = create_complete_bipartite_graph(N)  # 简单双向图
    g = np.random.randint(0, 1, (N, N), dtype=np.int8)
    print(g.shape)
    g = np.array(g, dtype=np.int8)
    g = torch.from_numpy(g)

    # 初始化模型
    model = AA_TGCN(input_steps=T, output_steps=t, features_dim=C)

    # 测试输入
    x = torch.randn(B, T, N, C)

    # 前向传播
    model
    out = model(x, g, log_u, training=True)
    print(f"输入维度: {x.shape} -> 输出维度: {out.shape}")
    assert out.shape == (B, t, N, C), "维度验证失败！"
