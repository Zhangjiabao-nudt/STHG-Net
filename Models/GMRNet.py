import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MSRModule(nn.Module):
    """适应节点特征的记忆存储召回模块"""

    def __init__(self, node_dim: int, mem_depth: int = 10):
        super().__init__()
        self.node_dim = node_dim
        self.mem_depth = mem_depth

        # 节点特征变换
        self.fc_f = nn.Linear(node_dim, node_dim)
        self.fc_c = nn.Linear(node_dim, node_dim)

        # 图注意力机制
        self.attn = nn.MultiheadAttention(node_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, F_t: torch.Tensor, memory_bank: list) -> torch.Tensor:
        """ F_t: [B, T, N, D] """
        if len(memory_bank) == 0:
            return F_t

        # 记忆库拼接 [B, Mem, N, D]
        C_hist = torch.stack(memory_bank[-self.mem_depth:], dim=1)[:,0,:,:,:]
        # print(C_hist.shape)
        # 特征投影
        Q = self.fc_f(F_t)  # [B, T, N, D]
        K = self.fc_c(C_hist)  # [B, Mem, N, D]
        # print(Q.shape, K.shape, C_hist.shape, "MSR")
        # 多头注意力
        B, T, N, D = Q.shape
        _, Mem, _, _ = K.shape
        Q = Q.view(B * T, N, D)
        K = K.view(B * Mem, N, D)
        attn_out, _ = self.attn(Q, K, K)  # [B*T, N, D]
        attn_out = attn_out.view(B, T, N, D)

        # 残差连接
        return self.layer_norm(F_t + attn_out)


class LMRCell(nn.Module):
    """节点时序记忆单元"""

    def __init__(self, input_dim: int, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 时间门控
        self.time_conv = nn.Conv1d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=3,
            padding=1
        )

        # 图卷积层
        self.gconv = GCNConv(hidden_dim, hidden_dim)

        # 记忆模块
        self.msr = MSRModule(hidden_dim)

        # 输出门
        self.out_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, h_prev, c_prev, memory_bank, edge_index):
        """
        x: [B, N, D]
        h_prev: [B, N, H]
        edge_index: 图结构连接
        """
        # 时间特征融合
        batch_size, N, _ = x.shape
        combined = torch.cat([x, h_prev], dim=-1)  # [B, N, D+H]
        gates = self.time_conv(combined.permute(0, 2, 1)).permute(0, 2, 1)
        i_t, g_t, f_t, o_t = torch.split(gates, self.hidden_dim, dim=-1)

        i_t = torch.sigmoid(i_t)
        g_t = torch.tanh(g_t)
        f_t = torch.sigmoid(f_t)

        # 记忆存储召回
        c_next = i_t * g_t + self.msr(f_t.unsqueeze(1), memory_bank).squeeze(1)

        # 图卷积传播
        batch_size = x.size(0)
        c_graph = c_next.reshape(-1, self.hidden_dim)  # [B*N, H]
        c_graph = F.relu(self.gconv(c_graph, edge_index))
        c_next = c_graph.view(batch_size, self.num_nodes, self.hidden_dim)

        # 输出门
        h_next = o_t * torch.tanh(c_next)
        # print(h_next.shape,c_next.shape, "LMRCell output")

        return h_next, c_next


class GMRNet(nn.Module):
    """适应图时序数据的GMR网络"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_nodes: int,
                 num_layers: int = 2,
                 pred_steps: int = 12,
                 edge_index: torch.Tensor = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_steps = pred_steps
        self.edge_index = edge_index
        self.hidden_dim = hidden_dim

        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # LMR层堆叠
        self.cells = nn.ModuleList([
            LMRCell(
                input_dim=hidden_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                num_nodes=num_nodes
            ) for i in range(num_layers)
        ])

        # 跨层连接
        self.fast_track = nn.ModuleList([
            nn.Linear((i + 1) * hidden_dim, hidden_dim)
            for i in range(num_layers - 1)
        ])

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, T_hist, N, D]
        Output: [B, T_pred, N, D]
        """
        B, T_hist, N, D = x.shape

        # 编码历史序列
        encoded = self.encoder(x)  # [B, T, N, H]

        # 初始化隐藏状态
        h_states = [torch.zeros(B, N, self.hidden_dim).to(x.device)
                    for _ in range(len(self.cells))]
        c_states = [torch.zeros_like(h) for h in h_states]

        memory_bank = []
        predictions = []

        # 递归预测
        for t in range(self.pred_steps):
            # 选择输入（训练时使用真实值）
            if t < T_hist and self.training:
                current_input = encoded[:, t]
            else:
                current_input = encoded[:, -1] if t == 0 else next_input

            layer_outputs = []
            for l, cell in enumerate(self.cells):
                # 跨层连接
                if l > 0:
                    fast_feat = self.fast_track[l - 1](
                        torch.cat(layer_outputs[:l], dim=-1)
                    )
                    current_input = current_input + fast_feat
                # print("{} layer".format(l), current_input.shape)
                # LMR单元处理
                h_next, c_next = cell(
                    x=current_input,
                    h_prev=h_states[l],
                    c_prev=c_states[l],
                    memory_bank=memory_bank,
                    edge_index=self.edge_index
                )
                # print(h_next.shape, c_next.shape, "GMR")
                # 更新状态
                h_states[l] = h_next
                c_states[l] = c_next
                layer_outputs.append(h_next)
                current_input = h_next

            # 更新记忆库
            memory_bank.append(c_next.detach().unsqueeze(1))
            if len(memory_bank) > 10:
                memory_bank.pop(0)

            # 生成预测
            pred = self.decoder(current_input)  # [B, N, D]
            predictions.append(pred.unsqueeze(1))

            # 准备下一时间步输入
            next_input = self.encoder(pred.unsqueeze(1))[:, 0]  # [B, N, H]

        return torch.cat(predictions, dim=1)  # [B, T_pred, N, D]


# 测试用例
def test_gmrnet():
    B, T_hist, N, D = 4, 24, 100, 8  # 批大小4，历史24步，100个节点，8维特征
    pred_steps = 12
    edge_index = torch.randint(0, N, (2, 200))  # 随机生成图连接

    model = GMRNet(
        input_dim=D,
        hidden_dim=64,
        num_nodes=N,
        pred_steps=pred_steps,
        edge_index=edge_index
    )

    # 前向测试
    x = torch.randn(B, T_hist, N, D)
    output = model(x)
    assert output.shape == (B, pred_steps, N, D), \
        f"Shape mismatch: {output.shape} vs expected {(B, pred_steps, N, D)}"

    # 梯度测试
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    loss.backward()

    print("测试通过！输入输出维度匹配，梯度计算正常")


if __name__ == "__main__":
    test_gmrnet()
    B, T_hist, N, D = 4, 24, 100, 8  # 批大小4，历史24步，100个节点，8维特征
    pred_steps = 12
    edge_index = torch.randint(0, N, (2, 200))  # 随机生成图连接

    model = GMRNet(
        input_dim=D,
        hidden_dim=64,
        num_nodes=N,
        pred_steps=pred_steps,
        edge_index=edge_index
    )

    # 前向测试
    x = torch.randn(B, T_hist, N, D)
    output = model(x)
    assert output.shape == (B, pred_steps, N, D), \
        f"Shape mismatch: {output.shape} vs expected {(B, pred_steps, N, D)}"

    # 梯度测试
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    loss.backward()

    print("测试通过！输入输出维度匹配，梯度计算正常")