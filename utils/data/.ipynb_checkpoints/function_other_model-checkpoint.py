import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


"""用于OVPGCN的距离阈值邻接矩阵生成"""
def generate_static_adj(valid_indices, mask_list, file_list, file_path, seq_len):
    """优化后的静态邻接矩阵生成函数"""
    for daily, index in enumerate(tqdm(valid_indices, desc='距离阈值邻接矩阵生成')):
        data_NT = mask_list[daily].transpose(0, 1)
        adj_path = f'{file_path}/{file_list[index]}.pth'

        # 生成稀疏邻接矩阵
        adj_sparse = generate_adjacency_matrix_optimized(data_NT)

        # 转换为PyTorch稀疏张量并保存
        sparse_tensor = sparse_coo_to_torch(adj_sparse)
        torch.save(sparse_tensor.coalesce(), adj_path)  # coalesce()用于合并重复索引


def generate_adjacency_matrix_optimized_mask(matrix):
    """基于坐标哈希和稀疏矩阵的优化实现"""
    # 提取有效点坐标 (y, x)
    points = np.argwhere(np.asarray(matrix) == 1)
    n = points.shape[0]
    if n == 0:
        return sp.coo_matrix((0, 0), dtype=np.float32)

    # ===== 阶段1：构建邻接矩阵 =====
    # 创建坐标到索引的哈希映射
    coord_to_idx = {(y, x): i for i, (y, x) in enumerate(points)}

    # 定义8邻域方向
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    # 收集所有边
    edges = []
    for i, (y, x) in enumerate(points):
        for dy, dx in directions:  # 注意坐标顺序
            ny, nx = y + dy, x + dx
            if (ny, nx) in coord_to_idx:
                j = coord_to_idx[(ny, nx)]
                if i != j:  # 排除自环
                    edges.append((i, j))

    # ===== 阶段2：构建稀疏矩阵 =====
    if not edges:
        return sp.coo_matrix((n, n), dtype=np.float32)

    # 转换为COO格式并去重
    rows, cols = zip(*edges)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

    # 合并重复边并二值化（确保值为1）
    adj = adj.tocsr()
    adj.data[:] = 1.0
    adj = adj.tocoo()

    # ===== 阶段3：对称归一化 =====
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5, where=(deg != 0))
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0

    # 应用归一化
    rows, cols = adj.row, adj.col
    norm_data = deg_inv_sqrt[rows] * deg_inv_sqrt[cols] * adj.data

    return sp.coo_matrix((norm_data, (rows, cols)), shape=adj.shape)


def generate_adjacency_matrix_optimized(matrix):
    """生成包含所有节点的稀疏邻接矩阵（基于网格8邻域）"""
    H, W = matrix.shape
    total_nodes = H * W

    # 生成所有节点坐标的网格
    i, j = np.indices((H, W))
    indices = i * W + j  # 行优先索引映射

    # 预计算所有可能的8邻域偏移
    di = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
    dj = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)

    # 向量化计算所有邻居坐标
    ni = (i[:, :, None] + di).reshape(H, W * 8)
    nj = (j[:, :, None] + dj).reshape(H, W * 8)

    # 构建有效掩码（边界检查）
    valid_mask = (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)

    # 计算源节点和目标节点索引
    src = np.repeat(indices, 8).reshape(H, W * 8)[valid_mask]
    dst = (ni[valid_mask] * W + nj[valid_mask])

    # 构建COO格式稀疏矩阵
    data = np.ones_like(src, dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(total_nodes, total_nodes))

    # 去重并二值化（CSR格式高效处理）
    adj = adj.tocsr()
    adj.data[:] = 1.0
    adj = adj.tocoo()

    # 对称归一化处理
    deg = adj.sum(axis=1).A.flatten()
    deg_inv_sqrt = np.power(deg, -0.5, out=np.zeros_like(deg), where=(deg != 0))

    rows, cols = adj.row, adj.col
    norm_data = deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

    return sp.coo_matrix((norm_data, (rows, cols)), shape=adj.shape)


def sparse_coo_to_torch(sparse_coo):
    """将SciPy COO矩阵转换为PyTorch稀疏张量"""
    rows = torch.from_numpy(sparse_coo.row.astype(np.int64))
    cols = torch.from_numpy(sparse_coo.col.astype(np.int64))
    values = torch.from_numpy(sparse_coo.data.astype(np.float32))
    indices = torch.stack([rows, cols], dim=0)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=sparse_coo.shape
    )


import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


def pearson_correlation_sparse(x: np.ndarray, threshold: float, chunk_size=200) -> sp.coo_matrix:
    """稀疏化皮尔逊相关系数计算 (分块处理优化内存)"""
    N, T = x.shape
    x_centered = x - x.mean(axis=1, keepdims=True)
    std = np.sqrt(np.sum(x_centered ** 2, axis=1) / (T - 1))

    rows, cols, data = [], [], []
    for i in tqdm(range(0, N, chunk_size), desc='分块计算相关系数'):
        chunk = x_centered[i:i + chunk_size]
        cov_block = chunk @ x_centered.T / (T - 1)
        std_block = std[i:i + chunk_size]

        # 向量化阈值过滤
        for k in range(chunk.shape[0]):
            global_idx = i + k
            corr_row = cov_block[k] / (std_block[k] * std + 1e-8)

            # 使用numpy向量化操作筛选
            mask = (np.abs(corr_row) >= threshold) & (np.arange(N) != global_idx)
            valid_idx = np.where(mask)[0]

            # 收集非零元素
            rows.extend([global_idx] * len(valid_idx))
            cols.extend(valid_idx.tolist())
            data.extend(corr_row[valid_idx].tolist())

    # 构建对称稀疏矩阵并去重
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    adj = adj.maximum(adj.T)  # 确保对称性
    adj = adj.tocsr()
    adj.data[:] = 1.0  # 二值化
    return adj.tocoo()


def sparse_normalization(adj: sp.coo_matrix) -> sp.coo_matrix:
    """稀疏矩阵对称归一化优化"""
    # 计算度矩阵
    deg = np.array(adj.sum(axis=1)).flatten()

    # 避免除零错误
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    np.divide(1.0, np.sqrt(deg, where=(deg != 0)), out=deg_inv_sqrt, where=(deg != 0))

    # 应用归一化
    rows, cols = adj.row, adj.col
    norm_data = deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

    return sp.coo_matrix((norm_data, (rows, cols)), shape=adj.shape)


def generate_pearson_adj(sst_mmap, valid_indices, file_list, file_path, seq_len, threshold=0.8):
    """优化后的皮尔逊邻接矩阵生成"""
    for index in tqdm(valid_indices, desc='皮尔逊邻接生成'):
        # 数据切片 (200x200区域)
        data = sst_mmap[index:index + seq_len, 600:800, 400:600]
        N = 200 * 200
        data = data.reshape(-1, N).T  # [N, T]

        # 生成稀疏邻接矩阵
        adj = pearson_correlation_sparse(data, threshold=threshold)
        norm_adj = sparse_normalization(adj)

        # 转换为PyTorch稀疏张量并保存
        adj_path = f'{file_path}/{file_list[index]}.pth'
        rows = torch.from_numpy(norm_adj.row.astype(np.int64))
        cols = torch.from_numpy(norm_adj.col.astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        sparse_tensor = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols]),
            values=values,
            size=norm_adj.shape
        )
        torch.save(sparse_tensor.coalesce(), adj_path)


import numpy as np
import scipy.sparse as sp
import torch
import h5py
from tqdm import tqdm


def pearson_correlation_streaming(x: np.ndarray, threshold: float,
                                  chunk_size=500, temp_file='temp_edges.h5'):
    """流式皮尔逊相关系数计算 (内存优化版)"""
    N, T = x.shape
    x_centered = x - x.mean(axis=1, keepdims=True)
    std = np.sqrt(np.sum(x_centered ** 2, axis=1) / np.sqrt(T - 1))

    # 创建HDF5文件存储临时结果
    with h5py.File(temp_file, 'w') as f:
        edge_group = f.create_group('edges')
    edge_group.create_dataset('rows', (0,), maxshape=(None,), dtype=np.int32)
    edge_group.create_dataset('cols', (0,), maxshape=(None,), dtype=np.int32)

    for i in tqdm(range(0, N, chunk_size), desc='流式计算相关系数'):
        chunk = x_centered[i:i + chunk_size].astype(np.float32)
    cov_block = np.dot(chunk, x_centered.T) / (T - 1)
    std_block = std[i:i + chunk_size]

    # 逐行处理避免内存峰值
    batch_rows, batch_cols = [], []
    for k in range(chunk.shape[0]):
        global_idx = i + k
    corr_row = cov_block[k] / (std_block[k] * std + 1e-8)

    # 向量化筛选
    mask = (np.abs(corr_row) >= threshold) & (np.arange(N) != global_idx)
    cols = np.where(mask)[0]
    rows = np.full_like(cols, global_idx)

    batch_rows.append(rows)
    batch_cols.append(cols)

    # 增量写入磁盘
    if batch_rows:
        batch_rows = np.concatenate(batch_rows)
    batch_cols = np.concatenate(batch_cols)

    # 扩展存储空间
    current_size = edge_group['rows'].shape[0]
    new_size = current_size + len(batch_rows)

    edge_group['rows'].resize((new_size,))
    edge_group['cols'].resize((new_size,))

    edge_group['rows'][current_size:new_size] = batch_rows
    edge_group['cols'][current_size:new_size] = batch_cols

    del chunk, cov_block, batch_rows, batch_cols  # 及时释放内存

    # 从磁盘加载边数据并构建对称矩阵
    with h5py.File(temp_file, 'r') as f:
        rows = f['edges/rows'][:]
    cols = f['edges/cols'][:]

    # 添加对称边并去重
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])

    # 创建COO矩阵并去重
    adj = sp.coo_matrix((np.ones_like(sym_rows), (sym_rows, sym_cols)), shape=(N, N))
    adj.sum_duplicates()
    adj.data[:] = 1.0  # 确保二值化

    return adj


def sparse_normalization_disk(adj: sp.coo_matrix, block_size=10000) -> sp.coo_matrix:
    """分块归一化处理 (内存敏感型操作)"""
    N = adj.shape[0]
    rows, cols = adj.row, adj.col
    data = adj.data

    # 分块计算度矩阵
    deg = np.zeros(N, dtype=np.float32)
    for i in tqdm(range(0, N, block_size), desc='分块计算度矩阵'):
        block = adj[i:i + block_size].tocsr()
        deg[i:i + block_size] = block.sum(axis=1).A1

    # 计算归一化系数
    deg_inv_sqrt = np.zeros_like(deg)
    valid = deg != 0
    deg_inv_sqrt[valid] = 1.0 / np.sqrt(deg[valid])

    # 分块应用归一化
    norm_data = np.empty_like(data)
    for i in tqdm(range(0, len(data), block_size), desc='分块归一化'):
        chunk = slice(i, i + block_size)
        norm_data[chunk] = deg_inv_sqrt[rows[chunk]] * deg_inv_sqrt[cols[chunk]]

    return sp.coo_matrix((norm_data, (rows, cols)), shape=adj.shape)


def generate_pearson_adj_optimized(sst_mmap, valid_indices, file_list, file_path, seq_len, threshold=0.8):
    """终极优化版皮尔逊邻接生成"""
    for index in tqdm(valid_indices, desc='邻接矩阵生成'):
        # 数据切片 (200x200区域)
        data = sst_mmap[index:index + seq_len, 600:800, 400:600]
        N = 200 * 200
        data = data.reshape(-1, N).T  # [N, T]

        # 流式生成邻接矩阵
        adj = pearson_correlation_streaming(data, threshold=threshold)
        norm_adj = sparse_normalization_disk(adj)

        # 直接保存为稀疏格式
        adj_path = f'{file_path}/{file_list[index]}.pth'
        rows = torch.from_numpy(norm_adj.row.astype(np.int64))
        cols = torch.from_numpy(norm_adj.col.astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))

        sparse_tensor = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols]),
            values=values,
            size=norm_adj.shape
        )
        torch.save(sparse_tensor.coalesce(), adj_path)

