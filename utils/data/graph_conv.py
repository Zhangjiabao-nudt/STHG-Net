import numpy as np
import scipy.sparse as sp
import torch
import copy
from torch.nn.functional import normalize


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


# def calculate_laplacian_with_self_loop_seq_len(matrixAll, seq_len):
#     result = []
#     for i in range(seq_len):
#         matrix = matrixAll[i]
#         matrix = matrix + torch.eye(matrix.size(0))
#         row_sum = matrix.sum(1)
#         d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
#         d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
#         d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
#         normalized_laplacian = (
#             matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
#         )
#         result.append(copy.deepcopy(normalized_laplacian))
#     return result