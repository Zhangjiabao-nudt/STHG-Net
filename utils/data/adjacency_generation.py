import scipy.sparse as sp
import torch
import numpy as np
import h5py
from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
from joblib import Parallel, delayed
import os
from tqdm import tqdm



def grid_hypergraph_to_graph(grid_data, E, m=True, random_state=None):
    """
    处理三维格点数据的超图转换

    Parameters:
    grid_data : np.ndarray (3D)
        格点矩阵，形状为（X,Y,features），非全nan的坐标视为有效顶点
    E : np.ndarray[object]
        超边集合，每个元素是包含有效坐标元组(x,y)的列表
    m : bool
        是否使用中介节点
    random_state : int
        随机种子

    Returns:
    torch.sparse.FloatTensor
        稀疏邻接矩阵（对称归一化）
    """
    # ===== 1. 顶点属性预处理 =====
    valid_mask = ~np.any(np.isnan(grid_data), axis=2)
    # print(valid_mask.shape)
    rows, cols = np.where(valid_mask)
    # valid_coords = np.column_stack((rows, cols))
    valid_features = grid_data[rows, cols, :]
    # print(valid_features.shape)
    grid_cols = grid_data.shape[1]
    n_nodes = len(rows)
    print(len(rows))
    # ===== 2. 超边快速转换 =====
    max_hash = (rows * grid_cols + cols).max()
    hash_table = np.full(max_hash + 1, -1, dtype=int)
    hash_table[rows * grid_cols + cols] = np.arange(n_nodes)

    # ===== 3. 超边向量化转换 =====
    edge_hashes = np.concatenate([np.array(e, dtype=int)[:, 0] * grid_cols +
                                  np.array(e, dtype=int)[:, 1] for e in E])
    edge_indices = hash_table[edge_hashes]

    split_points = np.cumsum([len(e) for e in E])[:-1]
    edge_indices = np.split(edge_indices, split_points)


    # ===== 4. 基于属性的投影优化 =====
    if random_state is not None:
        np.random.seed(random_state)
    rv = np.random.randn(valid_features.shape[1])

    # ===== 5. 修正后的边生成逻辑 =====
    rows_idx, cols_idx, data = [], [], []
    count = 0
    for indices in edge_indices:
        if len(indices) < 2:
            print(count)
            count+=1
            continue

        feats = valid_features[indices]
        projs = feats @ rv
        s, i = np.argmax(projs), np.argmin(projs)
        Se, Ie = indices[s], indices[i]
        k = len(indices)
        c = 2 * k - 3 if m else k

        # 基础边（保持双向）
        rows_idx.extend([Se, Ie])
        cols_idx.extend([Ie, Se])
        data.extend([1 / c] * 2)

        # 修正的中介边生成（关键修改部分）
        if m and k > 2:
            mask = np.ones(k, dtype=bool)
            mask[[s, i]] = False
            mediators = indices[mask]

            # 生成Se的双向中介边
            se_src = np.concatenate([np.full_like(mediators, Se), mediators])
            se_dst = np.concatenate([mediators, np.full_like(mediators, Se)])

            # 生成Ie的双向中介边
            ie_src = np.concatenate([np.full_like(mediators, Ie), mediators])
            ie_dst = np.concatenate([mediators, np.full_like(mediators, Ie)])

            # 合并所有边
            rows_idx.extend(np.concatenate([se_src, ie_src]).tolist())
            cols_idx.extend(np.concatenate([se_dst, ie_dst]).tolist())
            data.extend([1 / c] * (4 * len(mediators)))  # 每个mediator生成4条边

    # ===== 6. 构建邻接矩阵 =====
    adj = sp.coo_matrix((data, (rows_idx, cols_idx)), shape=(n_nodes, n_nodes))
    adj.sum_duplicates()

    # 添加自环（权重1.0）
    adj = adj + sp.eye(n_nodes, format='coo')  # 正确添加自环

    # 对称归一化
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    adj = adj.multiply(d_inv_sqrt).T.multiply(d_inv_sqrt).tocoo()

    # ===== 7. 转换为PyTorch张量 =====
    indices = torch.FloatTensor(np.vstack([adj.row, adj.col]))
    return torch.sparse_coo_tensor(indices, adj.data, adj.shape)



def normalize_adj(adj):
    """对称归一化邻接矩阵: D^(-1/2) (A + I) D^(-1/2)"""
    # 添加自环 (Add self-loops)
    print("1")
    adj = adj + np.eye(adj.shape[0])

    # 计算度矩阵 (Compute degree matrix)
    row_sum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    print("n")
    # 对称归一化 (Symmetric normalization)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    print("zero")
    return norm_adj


# ----------------------------
# 修改后的核心函数：增加块内进度条
# ----------------------------
# def compute_block(args):
#     """每个进程写入独立的临时HDF5文件（带进度条）"""
#     block_id, sst_data, adj, block_size, output_dir = args
#     N = sst_data.shape[0]
#     start = block_id * block_size
#     end = min((block_id + 1) * block_size, N)
#
#     # 为每个块创建独立文件
#     block_path = os.path.join(output_dir, f"block_{block_id:04d}.h5")
#     with h5py.File(block_path, 'w') as f:
#         block_data = f.create_dataset('C_block', (end - start, N), dtype='float32')
#
#         # 块内进度条（添加desc参数标识不同块）
#         pbar = tqdm(
#             total=end - start,
#             desc=f"Block {block_id:03d}",
#             position=block_id % 10,  # 限制同时显示进度条数量
#             leave=False
#         )
#
#         for local_i, global_i in enumerate(range(start, end)):
#             sst_i = sst_data[global_i].reshape(-1, 1)
#             row = np.full(N, np.inf, dtype='float32')
#
#             # 处理相邻节点
#             neighbors = np.where(adj[global_i] == 1)[0]
#             for j in neighbors:
#                 if global_i == j:
#                     row[j] = 0.0
#                 else:
#                     sst_j = sst_data[j, :].reshape(-1, 1)
#                     distance, _ = fastdtw(sst_i, sst_j, radius=5, dist=euclidean)
#                     row[j] = distance
#
#             block_data[local_i] = row
#             pbar.update(1)  # 更新进度
#
#         pbar.close()


# ----------------------------
# 主函数：增强合并过程的进度显示
# ----------------------------
# def generate_static_graph(sst_data, adj, output_dir='/tmp/blocks', block_size=500, n_jobs=64):
#     """
#     输入:
#         sst_data: N*T 的 numpy数组
#         adj: N*N的空间邻接矩阵
#         output_dir: 临时块文件存储目录
#         block_size: 每块处理的行数
#     输出:
#         合并后的邻接矩阵 A_s (HDF5文件)
#     """
#     # 创建临时目录（带进度提示）
#     print(f"🛠️ 创建临时目录: {output_dir}")
#     os.makedirs(output_dir, exist_ok=True)
#     N, T = sst_data.shape
#
#     # 验证邻接矩阵（带进度提示）
#     print("🔍 验证邻接矩阵格式...")
#     assert adj.shape == (N, N), "邻接矩阵维度不匹配"
#     assert np.all(adj.diagonal() == 0), "邻接矩阵应无自环"
#
#     # 并行计算所有块（主进度条）
#     print("🚀 启动并行计算...")
#     n_blocks = (N + block_size - 1) // block_size
#     Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
#         delayed(compute_block)((bid, sst_data, adj, block_size, output_dir))
#         for bid in tqdm(range(n_blocks), desc="总进度", position=0)
#     )
#
#     # 合并所有块文件（带详细进度）
#     print("🔗 合并临时文件...")
#     merged_path = os.path.join(output_dir, 'merged_result.h5')
#     with h5py.File(merged_path, 'w') as h5_merged:
#         C = h5_merged.create_dataset('C', (N, N), dtype='float32')
#
#         # 合并进度条（显示速度）
#         with tqdm(total=n_blocks, desc="合并进度", unit="block", position=0) as pbar:
#             for block_id in range(n_blocks):
#                 block_path = os.path.join(output_dir, f"block_{block_id:04d}.h5")
#                 with h5py.File(block_path, 'r') as h5_block:
#                     block_data = h5_block['C_block'][:]
#                     start = block_id * block_size
#                     end = min((block_id + 1) * block_size, N)
#                     C[start:end] = block_data
#                 pbar.update(1)
#                 pbar.set_postfix({"当前块": f"{block_id:04d}"})
#
#     # 归一化过程（带进度提示）
#     print("📊 生成最终邻接矩阵...")
#     with h5py.File(merged_path, 'r+') as f:
#         C = f['C']
#         valid_mask = (C[...] != np.inf)
#         valid_distances = C[valid_mask]
#
#         # 归一化进度条
#         with tqdm(total=5, desc="处理阶段", position=0) as pbar:
#             pbar.set_description("计算极值")
#             C_min = valid_distances.min() if len(valid_distances) > 0 else 0
#             C_max = valid_distances.max() if len(valid_distances) > 0 else 1
#             pbar.update(1)
#
#             pbar.set_description("归一化")
#             C_prime = np.full_like(C[...], np.inf)
#             if len(valid_distances) > 0:
#                 C_prime[valid_mask] = (valid_distances - C_min) / (C_max - C_min + 1e-8)
#             pbar.update(1)
#
#             pbar.set_description("指数转换")
#             A_s = np.exp(-(C_prime ** 2))
#             A_s[~valid_mask] = 0.0
#             pbar.update(1)
#
#             pbar.set_description("写入结果")
#             f.create_dataset('A_s', data=A_s.astype(np.float32))
#             pbar.update(2)
#
#     print("✅ 处理完成！")
#     return A_s


# if __name__ == "__main__":
    # # 生成3x3x2的格点数据，其中有效顶点为(0,0)和(1,1)
    # grid_data = np.full((3, 3, 2), np.nan)
    # grid_data[0, 0, :] = [1.0, 2.0]  # 有效顶点0
    # grid_data[1, 1, :] = [3.0, 4.0]  # 有效顶点1
    # print(grid_data)
    # # 定义一个超边，仅包含两个有效顶点
    # E = [[(0, 0), (1, 1)]]
    #
    #
    # a = np.array([1, np.nan, 2])
    #
    # # 转换为图结构
    # adj = grid_hypergraph_to_graph(
    #     grid_data=grid_data,
    #     E=E,
    #     m=True,
    #     random_state=42
    # )
    #
    # print(f"生成的邻接矩阵形状: {adj.shape}")
    # print(f"非零元素数量: {adj._nnz()}")
# ----------------------------
# 使用示例（增加执行时间预估）
# ----------------------------

if __name__ == "__main__":
    # 生成示例数据（带进度提示）
    print("📦 准备示例数据...")
    N, T = 29021, 365
    dummy_sst = np.random.rand(N, T).astype(np.float32)

    # 加载邻接矩阵（带进度提示）
    print("🔗 加载空间邻接矩阵...")
    npz_file = '/home/jameszhang/桌面/SST/data/data_gradient/adj.npz'
    data = np.load(npz_file)
    adj_matrix = data['adj']

    # 执行生成器（带预估时间）
    print("⏳ 开始生成静态图（预计需要较长时间）...")
    # A_s = generate_static_graph(
    #     dummy_sst,
    #     adj=adj_matrix,
    #     output_dir='/home/jameszhang/path',
    #     block_size=500,
    #     n_jobs=10
    # )
    A_s = np.load('/home/jameszhang/path.npz')["adj"]
    A_s = normalize_adj(A_s)
    np.savez('/home/jameszhang/path_end.npz', adj=A_s)
    print("🎉 邻接矩阵已生成！")