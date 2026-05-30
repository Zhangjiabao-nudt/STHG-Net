import argparse
import torch
import torch.nn as nn

from Models.layers import TimePeriodicEncoder, AbsolutePositionalEncoder
from utils.data.graph_conv import calculate_laplacian_with_self_loop


class TGCNGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        # self.register_buffer(
        #     "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        # )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state, adj, adj_long=None):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = adj @ concatenation  # short

        a_times_concat = adj @ a_times_concat # long

        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class LSTGCNGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(LSTGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias

        # 用 Linear 层替代手动参数 ------------------------------------------
        self.fc_short = nn.Linear(
            in_features=num_gru_units + 1,  # 输入维度 = gru_units + 1
            out_features=output_dim,  # 输出维度
            bias=True  # 保留偏置
        )
        self.fc_long = nn.Linear(
            in_features=num_gru_units + 1,
            out_features=output_dim,
            bias=True
        )

        # 自适应注意力权重生成层（保持不变）
        self.attention = nn.Sequential(
            nn.Linear(2 * output_dim, 2),  # 输入为短期和长期输出的拼接
            nn.Softmax(dim=-1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化 Linear 层的权重和偏置 --------------------------------------
        nn.init.xavier_uniform_(self.fc_short.weight)
        nn.init.constant_(self.fc_short.bias, self._bias_init_value)
        nn.init.xavier_uniform_(self.fc_long.weight)
        nn.init.constant_(self.fc_long.bias, self._bias_init_value)

        # 初始化注意力层（保持不变）
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, inputs, hidden_state, adj, adj_long=None):
        batch_size, num_nodes = inputs.shape

        # 拼接输入和隐藏状态 [x, h] -----------------------------------------
        inputs = inputs.view(batch_size, num_nodes, 1)
        hidden_state = hidden_state.view(batch_size, num_nodes, self._num_gru_units)
        concatenation = torch.cat((inputs, hidden_state), dim=2)

        # 调整维度以进行图卷积 -----------------------------------------------
        concatenation = concatenation.permute(1, 2, 0).contiguous()  # [num_nodes, gru_units+1, batch_size]
        concatenation = concatenation.view(num_nodes, (self._num_gru_units + 1) * batch_size)

        # 计算短期邻接矩阵的输出 --------------------------------------------
        a_times_concat_short = adj @ concatenation
        a_times_concat_short = adj @ a_times_concat_short

        a_times_concat_short = a_times_concat_short.view(num_nodes, self._num_gru_units + 1, batch_size)
        a_times_concat_short = a_times_concat_short.permute(2, 0,
                                                            1).contiguous()  # [batch_size, num_nodes, gru_units+1]
        a_times_concat_short = a_times_concat_short.view(-1,
                                                         self._num_gru_units + 1)  # [batch_size * num_nodes, gru_units+1]
        outputs_short = self.fc_short(a_times_concat_short)  # 使用 Linear 层
        outputs_short = outputs_short.view(batch_size, num_nodes, self._output_dim)

        # 计算长期邻接矩阵的输出（如果提供 adj_long）-------------------------
        outputs_long = None
        if adj_long is not None:
            a_times_concat_long = adj_long @ concatenation
            a_times_concat_long = a_times_concat_long.view(num_nodes, self._num_gru_units + 1, batch_size)
            a_times_concat_long = a_times_concat_long.permute(2, 0, 1).contiguous()
            a_times_concat_long = a_times_concat_long.view(-1, self._num_gru_units + 1)
            outputs_long = self.fc_long(a_times_concat_long)  # 使用 Linear 层
            outputs_long = outputs_long.view(batch_size, num_nodes, self._output_dim)

        # 自适应加权混合（保持不变）------------------------------------------
        if outputs_long is not None:
            combined = torch.cat([outputs_short, outputs_long], dim=-1)
            attention_weights = self.attention(combined)
            outputs = (attention_weights[:, :, 0:1] * outputs_short +
                       attention_weights[:, :, 1:2] * outputs_long)
        else:
            outputs = outputs_short

        # 调整输出维度 -----------------------------------------------------
        outputs = outputs.view(batch_size, num_nodes * self._output_dim)
        return outputs


class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        # self.register_buffer("adj", adj)
        self.graph_conv1 = LSTGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = LSTGCNGraphConvolution(
            self._hidden_dim, self._hidden_dim
        )
        self.norm1 = nn.LayerNorm(self._hidden_dim)
        self.norm2 = nn.LayerNorm(self._hidden_dim*2)
    def forward(self, inputs, hidden_state, adj, long_adj):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state, adj, long_adj))

        # 残差连接

        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state, adj, long_adj))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state
        new_hidden_state_2 =  (1.0 - u) * c
        return new_hidden_state+new_hidden_state_2, new_hidden_state+new_hidden_state_2

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, adj_shape, hidden_dim: int, d_model:int, seq_len:int, **kwargs):
        super(TGCN, self).__init__()
        print('Tgcn', adj_shape)
        self._input_dim = adj_shape
        self._hidden_dim = hidden_dim
        # self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self._input_dim, self._hidden_dim)
        # self.periodic_encoder = TimePeriodicEncoder(d_model=d_model, period_freqs=[seq_len])
        self.abs_encoder = AbsolutePositionalEncoder(d_model=self._input_dim)
        # 特征融合层
        # self.feature_fc = nn.Linear(self._input_dim + d_model, self._hidden_dim)

        # 新增自适应权重层
        self.weight_fc = nn.Linear(1, 1)  # 用于生成每个时间步的权重
    def forward(self, inputs, adj, long_adj=None):
        batch_size, seq_len, num_nodes = inputs.shape

        # t_indices  = torch.arange(0, seq_len).repeat(batch_size, 1)
        # periodic_pe = self.periodic_encoder(t_indices)
        # 拼接原始特征与编码
        # x_combined = torch.cat([inputs, periodic_pe], dim=-1)  # (batch, T, N + d_model)
        # 特征融合
        # x_fused = self.feature_fc(x_combined)  # (batch, T, hidden_dim)

        # 添加绝对位置编码
        x_fused = self.abs_encoder(inputs)  # (batch, T, hidden_dim)


        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            # output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state, adj)
            output, hidden_state = self.tgcn_cell(x_fused[:, i, :], hidden_state, adj, long_adj)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))

        return output
        # outputs_list = []
        # for i in range(seq_len):
        #     output, hidden_state = self.tgcn_cell(x_fused[:, i, :], hidden_state, adj, long_adj)
        #     output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        #     outputs_list.append(output)
        # # 2. 堆叠为四维张量 (batch_size, seq_len, num_nodes, hidden_dim)
        # outputs = torch.stack(outputs_list, dim=1)
        # del outputs_list
        # # 3. 自适应加权融合
        # # 计算每个时间步的权重分数
        # pooled = outputs.mean(dim=3)  # 在hidden_dim维度取平均 (batch, seq, nodes)
        # scores = self.weight_fc(pooled.unsqueeze(-1)).squeeze(-1)  # (batch, seq, nodes)
        #
        # # 归一化权重 (每个节点的所有时间步权重和为1)
        # weights = torch.softmax(scores, dim=1)  # (batch, seq, nodes)
        # weights = weights.unsqueeze(-1)  # 扩展维度 (batch, seq, nodes, 1)
        #
        # # 加权融合得到最终输出
        # final_output = (outputs * weights).sum(dim=1)  # (batch, nodes, hidden_dim)
        #
        # return final_output


    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--d_model", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
