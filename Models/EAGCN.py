# import argparse
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data.remote_backend_utils import num_nodes
#
#
# class StaticGraphGenerator:
#     """静态图生成器（预计算，非训练参数）"""
#
#     @staticmethod
#     def compute_dtw_matrix(sst_sequences):
#         """Params:
#             sst_sequences: [N, T], N区域数，T时间步长
#         Returns:
#             A_s: [N, N], 静态邻接矩阵"""
#         # 实际应用需替换为高效DTW实现（如fastdtw库）
#         N = sst_sequences.shape[0]
#         A_s = torch.zeros((N, N))
#         for i in range(N):
#             for j in range(N):
#                 dist = torch.norm(sst_sequences[i] - sst_sequences[j], p=2)
#                 A_s[i, j] = torch.exp(-dist ** 2)
#         return A_s
#
#
# class EAGC(nn.Module):
#     """增强自适应图卷积（公式9）"""
#
#     def __init__(self, num_nodes, in_dim, out_dim, embed_dim=64):
#         super().__init__()
#         # self.static_A = static_A  # [N, N]
#         self.embed_dim = embed_dim
#         self.N = num_nodes
#
#         # 动态图参数
#         self.Ea = nn.Parameter(torch.randn(self.N, embed_dim))  # 公式(5)
#         # 权重参数（公式9）
#         self.W_p = nn.Linear(embed_dim, in_dim, bias=False)
#         self.b_p = nn.Linear(embed_dim, out_dim, bias=False)
#
#     def forward(self, X, static_A):
#         """Params:
#             X: [B, N, F_in], B批次大小，N区域数，F_in输入特征维度
#         Returns:
#             Z: [B, N, F_out]"""
#         # 动态图生成（公式5）
#         B, N, F_in = X.shape
#         dynamic_A = F.softmax(F.relu(self.Ea @ self.Ea.T), dim=-1)  # [N, N]
#
#         # 融合静态图与动态图（公式8-9）
#         fused_A = torch.stack([static_A, dynamic_A], dim=0)  # [2, N, N]
#         fused_A = F.conv2d(fused_A.unsqueeze(0),
#                            weight=torch.ones(1, 2, 1, 1), padding=0).squeeze()
#         norm_A = F.softmax(fused_A, dim=-1) + torch.eye(self.N).to(X.device)
#         # print(X.shape, self.W_p(self.Ea).shape, "EAGC")
#
#         # print(X.shape, norm_A.shape, self.W_p(self.Ea).view(self.N, F_in, -1).shape, "EAGC")
#         # print(self.W_p(self.Ea).view(self.N, F_out, -1))
#         # print(torch.einsum('bni,nio->bno', X, self.W_p(self.Ea)).shape)
#         # 图卷积运算（公式9）
#         X_embed = torch.einsum('bni,nio->bno', X, self.W_p(self.Ea).view(self.N, F_in, -1))  # [B, N, F_out]
#         # print(X_embed.shape, "EAGC")
#         X_bias = self.b_p(self.Ea).unsqueeze(0).expand(X.shape[0], -1, -1)
#         Z = torch.bmm(norm_A.unsqueeze(0).expand(X.shape[0], -1, -1), X_embed) + X_bias
#         return Z
#
#
# class GGRUBlock(nn.Module):
#     """图门控循环单元（公式6）"""
#
#     def __init__(self, num_nodes, input_dim, hidden_dim):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         # 三个EAGC模块对应更新门z、重置门r和候选状态h
#         self.EAGC_z = EAGC(num_nodes, input_dim + hidden_dim, hidden_dim)
#         self.EAGC_r = EAGC(num_nodes, input_dim + hidden_dim, hidden_dim)
#         self.EAGC_h = EAGC(num_nodes, input_dim + hidden_dim, hidden_dim)
#
#     def forward(self, X, H_prev, static_A):
#         """Params:
#             X: [B, N, F_in], 当前时间步输入
#             H_prev: [B, N, F_hid], 前一时刻隐藏状态
#         Returns:
#             H_new: [B, N, F_hid], 新隐藏状态"""
#         # 拼接输入和隐藏状态（公式6）
#         XH = torch.cat([X, H_prev], dim=-1)  # [B, N, F_in + F_hid]
#         # print(XH.shape, H_prev.shape, "GGRUBlock")
#         # 计算更新门z和重置门r
#         z = torch.sigmoid(self.EAGC_z(XH, static_A))  # [B, N, F_hid]
#         r = torch.sigmoid(self.EAGC_r(XH, static_A))  # [B, N, F_hid]
#
#         # 计算候选状态h_tilde
#         h_tilde = torch.tanh(self.EAGC_h(torch.cat([X, r * H_prev], dim=-1), static_A))
#
#         # 更新隐藏状态（公式6最后一行）
#         H_new = z * H_prev + (1 - z) * h_tilde
#         return H_new
#
#     def init_hidden(self, batch_size, device):
#         """初始化隐藏状态为全零"""
#         return torch.zeros(batch_size, self.EAGC_z.N, self.hidden_dim).to(device)
#
#
# class MultiGranularEncoder(nn.Module):
#     """多时间粒度编码器（处理日、月、年分支）"""
#
#     def __init__(self, num_nodes, input_dims, hidden_dim, num_layers=2, times_len=3):
#         super().__init__()
#         self.num_branches = len(input_dims)
#         self.hidden_dim = hidden_dim
#
#         # 每个分支包含多层GGRUBlock
#         self.branches = nn.ModuleList([
#             nn.Sequential(*[
#                 GGRUBlock(num_nodes, input_dim if i==0 else hidden_dim, hidden_dim)
#                 for i in range(num_layers)
#             ]) for input_dim in input_dims
#         ])
#
#     def forward(self, X_daily, X_monthly, X_yearly, static_A):
#         """Params:
#             X_daily: [B, T_daily, N, F_daily]
#             X_monthly: [B, T_monthly, N, F_monthly]
#             X_yearly: [B, T_yearly, N, F_yearly]
#         Returns:
#             outputs: [B, N, 3*F_hid]"""
#         batch_size, _, N, _ = X_daily.shape
#         device = X_daily.device
#
#         # 存储各分支最终隐藏状态
#         branch_outputs = []
#
#         # 处理每个分支（日、月、年）
#         for branch_idx, (branch, X) in enumerate(zip(self.branches,
#                                                      [X_daily, X_monthly, X_yearly])):
#             # 初始化每层的隐藏状态
#             hidden_states = [layer.init_hidden(batch_size, device) for layer in branch]
#             # 按时间步迭代处理
#             for t in range(X.shape[1]):
#                 x_t = X[:, t, :, :]  # [B, N, F]
#                 # 逐层传递
#                 for layer_idx, (layer, h_prev) in enumerate(zip(branch, hidden_states)):
#                     h_new = layer(x_t, h_prev, static_A)
#                     hidden_states[layer_idx] = h_new  # 更新隐藏状态
#                     x_t = h_new  # 下一层的输入
#
#             branch_outputs.append(hidden_states[-1])  # 取最后一层输出
#
#         # 拼接三个分支的输出
#         return torch.cat(branch_outputs, dim=-1)  # [B, N, 3*F_hid]
#
#
# class EA_GCN(nn.Module):
#     """EA-GCN主模型"""
#
#     def __init__(self, num_nodes, daily_dim, monthly_dim, yearly_dim, pred_steps, hidden_dim=64, num_layers=2):
#         super().__init__()
#         self.encoder = MultiGranularEncoder(
#             num_nodes=num_nodes,
#             input_dims=[daily_dim, monthly_dim, yearly_dim],
#             hidden_dim=hidden_dim,
#             num_layers=num_layers,
#         )
#         # 解码器：时间维度卷积
#         self.decoder = nn.Sequential(
#             nn.Conv1d(3 * hidden_dim, pred_steps, kernel_size=3, padding=1),
#             nn.BatchNorm1d(pred_steps),
#         )
#         self.pred_steps = pred_steps
#         self.hidden_dim = hidden_dim
#     def forward(self, X_daily, X_monthly, X_yearly, static_A):
#         """Params:
#             X_daily: [B, T_daily, N, F_daily=1]
#             X_monthly: [B, T_monthly, N, F_monthly=1]
#             X_yearly: [B, T_yearly, N, F_yearly=1]
#         Returns:
#             output: [B, N, pred_steps]"""
#         encoded = self.encoder(X_daily, X_monthly, X_yearly, static_A)  # [B, N, 3*64]
#         # print(encoded.shape, "EA_GCN")
#         output = self.decoder(encoded.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, pred_steps]
#         return output
#
#     @staticmethod
#     def add_model_specific_arguments(parent_parser):
#         parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
#         # parser.add_argument("--daily_dim", type=int, default=30)
#         # parser.add_argument('monthly_dim', type=int, default=90)
#         # parser.add_argument('yearly_dim', type=int, default=360)
#         parser.add_argument('--hidden_dim', type=int, default=64)
#         parser.add_argument('--num_layers', type=int, default=2)
#         parser.add_argument("--pred_steps", type=int, default=7)
#         return parser
#
#     @property
#     def hyperparameters(self):
#         return {
#             'hidden_dim': self.hidden_dim,
#             'output_dim': self.pred_steps,
#         }
#
#
#
# # 示例用法
# if __name__ == "__main__":
#     # 预计算静态图（N=400）
#     N = 400
#     static_A = StaticGraphGenerator.compute_dtw_matrix(torch.randn(N, 365))
#     print(static_A.shape)
#     # 初始化模型
#     model = EA_GCN(
#         num_nodes=N,
#         daily_dim=1,  # 日粒度输入特征维度（例如过去7天）
#         monthly_dim=1,  # 月粒度（过去12个月均值）
#         yearly_dim=1,  # 年粒度（过去1年均值）
#         pred_steps=60  # 预测60天
#     )
#
#     # 示例输入（Batch=32）
#     X_daily = torch.randn(1, 7, N, 1)  # [B, T_daily=7, N, F_daily=7]
#     X_monthly = torch.randn(1, 12, N, 1)
#     X_yearly = torch.randn(1, 1, N, 1)
#
#     # 前向传播
#     pred = model(X_daily, X_monthly, X_yearly, static_A)  # 输出[B, N, 60]
#     print(pred.shape)  # torch.Size([32, 400, 60])


import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticGraphGenerator:
    """静态图生成器（预计算，非训练参数）"""

    @staticmethod
    def compute_dtw_matrix(sst_sequences):
        N = sst_sequences.shape[0]
        A_s = torch.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dist = torch.norm(sst_sequences[i] - sst_sequences[j], p=2)
                A_s[i, j] = torch.exp(-dist ** 2)
        return A_s


class EAGC(nn.Module):
    """增强自适应图卷积"""

    def __init__(self, num_nodes, in_dim, out_dim, embed_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.N = num_nodes

        self.Ea = nn.Parameter(torch.randn(self.N, embed_dim))
        self.W_p = nn.Linear(embed_dim, in_dim, bias=False)
        self.b_p = nn.Linear(embed_dim, out_dim, bias=False)

    def forward(self, X, static_A):
        B, N, F_in = X.shape
        start = time.time()
        dynamic_A = F.relu(self.Ea @ self.Ea.T)
        print(time.time() - start, 1)
        start = time.time()

        fused_A = torch.stack([static_A, dynamic_A], dim=0)
        fused_A = F.conv2d(fused_A.unsqueeze(0),
                           weight=torch.ones(1, 2, 1, 1), padding=0).squeeze()
        # fused_A = static_A + dynamic_A
        norm_A = F.softmax(fused_A, dim=-1) + torch.eye(self.N).to(X.device)
        print(time.time() - start, 2)
        start = time.time()
        X_embed = torch.einsum('bni,nio->bno', X, self.W_p(self.Ea).view(N, F_in, -1))
        print(time.time() - start, 3)
        start = time.time()
        X_bias = self.b_p(self.Ea).unsqueeze(0).expand(B, -1, -1)
        print(time.time() - start, 4)
        start = time.time()

        Z = torch.bmm(norm_A.unsqueeze(0).expand(B, -1, -1), X_embed) + X_bias
        print(time.time() - start, 5)
        return Z


# class EAGC(nn.Module):
#     """增强自适应图卷积（优化版）"""
#
#     def __init__(self, num_nodes, in_dim, out_dim, embed_dim=64):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.N = num_nodes
#
#         # 初始化参数时使用较小方差
#         self.Ea = nn.Parameter(torch.randn(self.N, embed_dim) * 0.1)
#
#         # 预注册单位矩阵（避免重复生成）
#         self.register_buffer("eye", torch.eye(self.N))
#
#         # 合并权重计算
#         self.W_p = nn.Linear(embed_dim, in_dim * out_dim, bias=False)
#         self.b_p = nn.Linear(embed_dim, out_dim, bias=False)
#
#     def forward(self, X, static_A):
#         B, N, F_in = X.shape
#
#         # 优化1：移除ReLU并使用更稳定的矩阵乘法
#         E = self.Ea / torch.norm(self.Ea, dim=1, keepdim=True)  # 特征归一化
#         dynamic_A = F.softmax(E @ E.T, dim=-1)  # [N, N]
#
#         # 优化2：简化图融合为加权求和
#         fused_A = static_A + dynamic_A
#
#         # 优化3：合并归一化与自连接操作
#         norm_A = F.softmax(fused_A, dim=-1) + self.eye.to(X.device)
#
#         # 优化4：使用更高效的矩阵乘法代替einsum
#         W = self.W_p(E).view(N, F_in, -1)  # [N, F_in, F_out]
#         X_embed = torch.bmm(X.transpose(0, 1), W).transpose(0, 1)  # [B, N, F_out]
#
#         # 优化5：利用广播机制代替显式expand
#         X_bias = self.b_p(E).unsqueeze(0)  # [1, N, F_out]
#
#         # 优化6：使用matmul自动广播代替bmm
#         return torch.matmul(norm_A.unsqueeze(0), X_embed) + X_bias  # [B, N, F_out]



class GGRUBlock(nn.Module):
    """图门控循环单元（公式6）"""

    def __init__(self, num_nodes, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 三个EAGC模块对应更新门z、重置门r和候选状态h
        self.EAGC_z = EAGC(num_nodes, input_dim + hidden_dim, hidden_dim)
        self.EAGC_r = EAGC(num_nodes, input_dim + hidden_dim, hidden_dim)
        self.EAGC_h = EAGC(num_nodes, input_dim + hidden_dim, hidden_dim)

    def forward(self, X, H_prev, static_A):
        """Params:
            X: [B, N, F_in], 当前时间步输入
            H_prev: [B, N, F_hid], 前一时刻隐藏状态
        Returns:
            H_new: [B, N, F_hid], 新隐藏状态"""
        # 拼接输入和隐藏状态（公式6）
        XH = torch.cat([X, H_prev], dim=-1)  # [B, N, F_in + F_hid]
        # print(XH.shape, H_prev.shape, "GGRUBlock")
        # 计算更新门z和重置门r
        z = torch.sigmoid(self.EAGC_z(XH, static_A))  # [B, N, F_hid]
        r = torch.sigmoid(self.EAGC_r(XH, static_A))  # [B, N, F_hid]

        # 计算候选状态h_tilde
        h_tilde = torch.tanh(self.EAGC_h(torch.cat([X, r * H_prev], dim=-1), static_A))

        # 更新隐藏状态（公式6最后一行）
        H_new = z * H_prev + (1 - z) * h_tilde
        return H_new, h_tilde

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态为全零"""
        return torch.zeros(batch_size, self.EAGC_z.N, self.hidden_dim).to(device)


class MultiGranularEncoder(nn.Module):
    """多时间粒度编码器（处理日、月、年分支）"""

    def __init__(self, num_nodes, input_dims, hidden_dim, num_layers=2, time_steps=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.linears = nn.Linear(input_dims * time_steps, hidden_dim)

        self.branches = nn.Sequential(*[
                # GGRUBlock(num_nodes, input_dims * time_steps if i ==0 else hidden_dim, hidden_dim)
                EAGC(num_nodes, hidden_dim , hidden_dim)
                for i in range(num_layers)])


    def forward(self, X_daily, X_monthly, X_yearly, static_A):
        B, T, N, F = X_daily.shape

        X_daily = X_daily.permute(0, 2, 1, 3).reshape(B, N, T * F)
        X_monthly = X_monthly.permute(0, 2, 1, 3).reshape(B, N, T * F)
        X_yearly = X_yearly.permute(0, 2, 1, 3).reshape(B, N, T * F)

        X_proj = torch.cat([X_daily, X_monthly, X_yearly], dim=-1)
        # device = X_daily.device
        X_proj = self.linears(X_proj)
        # hidden_states = None
        #   多层EAGC处理
        for layer in self.branches:
            # if hidden_states is None:
                # hidden_states = layer.init_hidden(B, device)
            start = time.time()
            # X_proj, hidden_states = layer(X_proj, hidden_states, static_A)
            X_proj = layer(X_proj, static_A)
            print(time.time() - start)
        # print(X_proj.size())
        return X_proj


class EA_GCN(nn.Module):
    """EA-GCN主模型"""

    def __init__(self, num_nodes,input_dims,
                 pred_steps, hidden_dim=64, num_layers=2, time_steps=3):
        super().__init__()
        self.encoder = MultiGranularEncoder(
            num_nodes=num_nodes,
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_steps=time_steps
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, pred_steps, kernel_size=3, padding=1),
            nn.BatchNorm1d(pred_steps),
        )
        self.pred_steps = pred_steps
        self.hidden_dim = hidden_dim

    def forward(self, X_daily, X_monthly, X_yearly, static_A):
        encoded = self.encoder(X_daily, X_monthly, X_yearly, static_A)
        output = self.decoder(encoded.permute(0, 2, 1)).permute(0, 2, 1)
        return output.permute(0, 2, 1)

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument("--pred_steps", type=int, default=7)
        return parser

    @property
    def hyperparameters(self):
        return {
            'hidden_dim': self.hidden_dim,
            'output_dim': self.pred_steps,
        }


if __name__ == "__main__":
    N = 400
    static_A = StaticGraphGenerator.compute_dtw_matrix(torch.randn(N, 365))

    model = EA_GCN(
        num_nodes=N,
        input_dims=7,
        pred_steps=60
    )

    X_daily = torch.randn(1, 7, N, 1)
    X_monthly = torch.randn(1, 7, N, 1)
    X_yearly = torch.randn(1, 7, N, 1)

    pred = model(X_daily, X_monthly, X_yearly, static_A)
    print(pred.shape)  # torch.Size([1, 400, 60])