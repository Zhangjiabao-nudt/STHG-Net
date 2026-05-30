import numpy as np
# from triton.language import dtype

from utils.data.cluster import HyperGraphConstructor
# from numba import jit
# from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from utils.data.adjacency_generation import grid_hypergraph_to_graph
from torch import Tensor


def calculate_temperature_gradient_time(sea_surface_temperature):
    # 假设 sea_surface_temperature 是一个三维数组，表示海表温度分布
    # 计算温度梯度，使用中心差分法 T , x, y
    gradient_3d = np.gradient(sea_surface_temperature, axis=0)
    # print(len(gradient_3d))
    print(gradient_3d.shape)
    return np.nanmean(gradient_3d, axis=0)


def calculate_temperature_gradient_spatio(sea_surface_temperature):
    """
    :param sea_surface_temperature: shape is [T， lat, lon]
    :return: dx,dy,
    """
    gradient_3d = np.gradient(sea_surface_temperature, edge_order=1)
    if np.isnan(gradient_3d[1]).all() and np.isnan(gradient_3d[2]).all():
        raise "all is nan"
    temperature_gradient_magnitude = np.nanmean(np.sqrt(gradient_3d[1] ** 2 + gradient_3d[2] ** 2), axis=0)
    return temperature_gradient_magnitude


# 年变化
def compute_temporal_gradient(sst_data, chunksize=100):
    """
    计算每个空间点的时间序列线性趋势梯度 长期变化率，年变化温度

    参数:
        sst_data (np.ndarray): 三维数组 [time, lat, lon]
        chunksize (int): 纬度分块大小

    返回:
        np.ndarray: 二维梯度数组 [lat, lon]
    """
    # 验证输入维度
    if sst_data.ndim != 3:
        raise ValueError("输入必须是三维数组 [time, lat, lon]")

    T, H, W = sst_data.shape

    # 生成时间序列
    x = np.arange(T)

    # 预计算常数项
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    denominator = T * sum_x2 - sum_x ** 2

    # 分块计算累加值
    gradient = np.zeros((H, W), dtype=np.float32)

    for lat_start in tqdm(range(0, H, chunksize), desc="Processing"):
        lat_end = min(lat_start + chunksize, H)
        chunk = sst_data[:, lat_start:lat_end, :]  # [T, chunk, W]

        # 计算块内累加
        sum_y = chunk.sum(axis=0)  # [chunk, W]
        sum_xy = (x[:, None, None] * chunk).sum(axis=0)  # [chunk, W]

        # 计算梯度
        chunk_grad = (T * sum_xy - sum_x * sum_y) / denominator
        gradient[lat_start:lat_end, :] = chunk_grad

    return gradient


# 日变化
def compute_instant_gradient(sst_data, chunksize=100):
    """
    基于中心差分计算平均时间梯度  短期温度变化，日变化
    参数:
        sst_data (np.ndarray): 三维数组 [time, lat, lon]
        chunksize (int): 经度分块大小
    返回:
        np.ndarray: 二维梯度数组 [lat, lon]
    """
    T, H, W = sst_data.shape

    # 初始化结果数组
    gradient = np.zeros((H, W), dtype=np.float32)

    # 分块处理
    for lon_start in tqdm(range(0, W, chunksize), desc="Processing"):
        lon_end = min(lon_start + chunksize, W)
        chunk = sst_data[:, :, lon_start:lon_end]  # [T, H, chunk]

        # 计算时间梯度
        time_grad = np.gradient(chunk, axis=0)  # [T, H, chunk]

        # 取时间平均
        chunk_avg = time_grad.mean(axis=0)  # [H, chunk]
        gradient[:, lon_start:lon_end] = chunk_avg

    return gradient


def is_leap(year):
    """判断是否为闰年"""
    if year % 4 != 0:
        return False
    elif year % 100 != 0:
        return True
    else:
        return year % 400 == 0


def compute_yearly_gradient(sst_mmap, data_dir, start_year, end_year, spatio_temporal=True):
    yearly_index = {}
    current_start = 0

    for year in range(start_year, end_year + 1):
        days = 365  # if is_leap(year) else 365
        end = current_start + days
        year_data = sst_mmap[current_start:end, 600:800, 400:600]
        yearly_index[str(year)] = np.array([current_start, end])
        current_start = end  # 更新下一年的起始索引
        year_data = np.where(year_data == 0, np.nan, year_data)
        if spatio_temporal:
            year_data = calculate_temperature_gradient_spatio(year_data)
            print(year_data.shape)
            print(np.sum(np.isnan(year_data)))
            np.save(f'{data_dir}/sst_{year}_data.npy', year_data)
        else:
            year_data = calculate_temperature_gradient_time(year_data)
            np.save(f'{data_dir}/sst_{year}.npy', year_data)
    # np.savez(f'{data_dir}/yearly_index.npz', **yearly_index)
    # 验证总天数是否正确
    # assert current_start == 10585, "索引总数与数据总量不符"


def compute_half_year_gradient(sst_mmap, data_dir, start_year, end_year):
    halfyearly_index = {}
    current_start = 0

    for year in range(start_year, end_year + 1):
        # 处理上半年（1-6月，182天）
        h1_start = current_start
        h1_end = current_start + 182
        h1_data = sst_mmap[h1_start:h1_end, 600:800, 400:600]
        halfyearly_index[f"{year}_H1"] = np.array([h1_start, h1_end])

        # 处理下半年（7-12月，183天）
        h2_start = h1_end
        h2_end = current_start + 365
        h2_data = sst_mmap[h2_start:h2_end, 600:800, 400:600]
        halfyearly_index[f"{year}_H2"] = np.array([h2_start, h2_end])

        # 更新索引到下一年的起始位置
        current_start += 365
        # 处理上半年数据
        h1_data = np.where(h1_data == 0, np.nan, h1_data)
        h1_grad = calculate_temperature_gradient_spatio(h1_data)
        np.save(f'{data_dir}/sst_{year}_H1_grad.npy', h1_grad)

        # 处理下半年数据
        h2_data = np.where(h2_data == 0, np.nan, h2_data)
        h2_grad = calculate_temperature_gradient_spatio(h2_data)
        np.save(f'{data_dir}/sst_{year}_H2_grad.npy', h2_grad)

    # 验证总天数
    # total_days = (end_year - start_year + 1) * 365
    # assert current_start == total_days, f"索引总数{current_start}与数据总量{total_days}不符"
    # np.savez(f'{data_dir}/half_year_index.npz', **halfyearly_index)

def compute_daily_gradient(sst_mmap, data_dir, valid_indices, seq_len, pred_len, file_list):
    max_mask = None
    for daily, index in enumerate(tqdm(valid_indices, desc='处理进度')):
        daily_data = sst_mmap[index:index + seq_len + pred_len, 600:800, 400:600]
        print("get origin data:", daily_data.shape, index, index + seq_len+pred_len)
        daily_data = np.where(daily_data == 0, np.nan, daily_data)
        daily_gradient = calculate_temperature_gradient_spatio(daily_data)
        print("compute gradient data", daily_gradient.shape)
        print(np.sum(np.isnan(daily_gradient)))
        if daily == 0:
            max_mask = ~np.isnan(daily_gradient)
        else:
            current_non_nan = ~np.isnan(daily_gradient)
            max_mask = max_mask & current_non_nan  # 逻辑与操作
        np.save(f'{data_dir}/sst_{file_list[index]}_gradient.npy', daily_gradient)

    # np.savez(f'{data_dir}/gradient_mask.npz', mask=max_mask)

def compute_and_save_graph(gradient_mmap, data_dir, valid_indices, file_list, max_mask):
    for daily, index in enumerate(tqdm(valid_indices, desc='超图生成')):
        # gradient_daily = np.where(max_mask, gradient_mmap[daily], np.nan)
        gradient_daily = np.where(max_mask[daily], gradient_mmap[daily], np.nan)
        print(np.sum(max_mask[daily]), np.sum(np.isnan(gradient_daily)))
        hyper_graph_constructor = HyperGraphConstructor(gradient_daily)
        hyper_graph_constructor.classify_sub_graphs()
        sub_graph = hyper_graph_constructor.get_sub_graphs()
        np.save(f'{data_dir}/{file_list[index]}.npy', np.array(sub_graph, dtype=object))

def compute_and_save_adjacency(sst_mmap, gradient_mmap, hyper_edges_list, data_dir, valid_indices, file_list, lens, feat, max_mask, use_normalize=False):
    for daily, index in enumerate(tqdm(valid_indices, desc='关联矩阵生成')):
        daily_data = sst_mmap[index:index + lens, 600:800, 400:600]

        hyper_edge = np.load(hyper_edges_list[daily], allow_pickle=True)
        # gradient_data = gradient_mmap[daily]
        # gradient_data = np.where(max_mask, gradient_mmap[daily], np.nan)
        gradient_data = np.where(max_mask[daily], gradient_mmap[daily], -1)
        print(np.sum(max_mask[daily]), np.sum(np.where(gradient_data==-1)))
        # 调整 daily_data 维度
        adjust_daily = np.transpose(daily_data, (1, 2, 0))  # 形状 (H, W, lens+1)
        if use_normalize:
            adjust_daily = adjust_daily - feat['mean'] / feat['std']
        # 扩展 gradient 维度
        gradient_expand = gradient_data[..., np.newaxis]  # 形状 (H, W, 1)
        # 沿时间轴拼接
        merge_data = np.concatenate([adjust_daily, gradient_expand], axis=2)  # 形状 (H, W, lens+2)
        adj = grid_hypergraph_to_graph(merge_data, hyper_edge, random_state=1)
        print(f"生成的邻接矩阵形状: {adj.shape}")
        torch.save(adj, f'{data_dir}/{file_list[index]}.pth')

def compute_mmap_statistics(mmap_path, metric_path, safe_chunk_mb=200):
    """
    计算内存映射文件的统计特征（max, min, mean, std）

    参数:
        mmap_path (str): 内存映射文件路径
        safe_chunk_mb (int): 每个块的最大内存占用(MB)，默认200MB

    返回:
        dict: 包含统计量的字典
    """
    # 加载内存映射文件
    mmap = np.load(mmap_path, mmap_mode='r')

    # 获取数组元信息
    dtype_size = mmap.dtype.itemsize  # 单元素字节数
    total_elements = mmap.size  # 总元素数
    array_shape = mmap.shape  # 数组维度

    # 计算合适的块大小
    elements_per_chunk = (safe_chunk_mb * 1024 ** 2) // dtype_size
    chunk_size = max(1, elements_per_chunk // np.prod(mmap.shape[1:]))

    print(f"数组维度: {array_shape}")
    print(f"内存占用: {total_elements * dtype_size / 1024 ** 2:.2f} MB")
    print(f"分块策略: 每块包含 {chunk_size} 个样本")

    # 初始化统计量（使用float64保证计算精度）
    global_max = -np.inf
    global_min = np.inf
    sum_total = 0.0
    sum_squares = 0.0
    processed_samples = 0

    # 分块处理数据
    with tqdm(total=array_shape[0], desc="处理进度") as pbar:
        for start_idx in range(0, array_shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, array_shape[0])

            # 读取当前块数据（自动内存映射）
            chunk = mmap[start_idx:end_idx, 600:800, 400:600]
            chunk = np.where(chunk == 0, np.nan, chunk)
            # 展平处理（保持内存效率）
            # flat_chunk = chunk.ravel()
            chunk = chunk[~np.isnan(chunk)].astype(np.float32)
            # 更新极值
            chunk_max = np.nanmax(chunk)
            chunk_min = np.nanmin(chunk)
            global_max = max(global_max, chunk_max)
            global_min = min(global_min, chunk_min)

            # 更新累加值（转换为float64防止溢出）
            chunk_float = chunk.astype(np.float64)
            sum_total += np.nansum(chunk_float)
            sum_squares += np.nansum(chunk_float ** 2)

            # 更新进度
            processed_samples += chunk.size
            pbar.update(end_idx - start_idx)

    # 计算最终统计量
    mean = sum_total / processed_samples
    std = np.sqrt((sum_squares / processed_samples) - (mean ** 2))
    np.savez(metric_path, mean=mean, std=std, max=global_max, min=global_min)
    print({
        "max": float(global_max),
        "min": float(global_min),
        "mean": float(mean),
        "std": float(std),
        "total_samples": processed_samples
    })


def pearson_correlation(x: Tensor) -> Tensor:
    """计算皮尔逊相关系数矩阵"""
    # 中心化数据 [N, T]
    x_centered = x - x.mean(dim=1, keepdim=True)

    # 计算协方差矩阵 [N, N]
    cov = torch.mm(x_centered, x_centered.T) / (x.size(1) - 1)

    # 计算标准差 [N]
    std = torch.sqrt(torch.sum(x_centered ** 2, dim=1) / (x.size(1) - 1))

    # 避免除零错误
    std_product = std[:, None] * std[None, :] + 1e-8

    # 计算相关系数矩阵 [N, N]
    corr_matrix = cov / std_product
    return corr_matrix


def build_adjacency_matrix(
        x: Tensor,
        threshold: float = 0.9,
        top_k: int = None,
        undirected: bool = True
) -> Tensor:
    """
    构建邻接矩阵
    Args:
        x: 节点时序数据 [num_nodes, num_timesteps]
        threshold: 相关系数阈值
        top_k: 每个节点保留的最大邻居数
        undirected: 是否生成无向图
    Returns:
        adj_matrix: 邻接矩阵 [num_nodes, num_nodes]
    """
    # 计算相关系数矩阵
    corr_matrix = pearson_correlation(x)

    # 应用绝对值阈值
    adj_matrix = (corr_matrix.abs() >= threshold).float()

    # 去除自环
    # adj_matrix.fill_diagonal_(0)

    # 应用top_k过滤
    if top_k is not None:
        # 获取每个节点的top_k邻居索引
        _, top_indices = torch.topk(
            corr_matrix.abs(),
            k=min(top_k + 1, corr_matrix.size(1)),  # +1考虑自身
            dim=1
        )

        # 创建mask矩阵
        mask = torch.zeros_like(adj_matrix)
        rows = torch.arange(corr_matrix.size(0)).unsqueeze(1)
        mask[rows, top_indices] = 1.0

        # 应用mask并保持对称性
        adj_matrix = adj_matrix * mask
        if undirected:
            adj_matrix = torch.maximum(adj_matrix, adj_matrix.T)
    del corr_matrix
    torch.cuda.empty_cache()
    adj_matrix = normalize_adj_TORCH(adj_matrix)
    return adj_matrix


def adjacency_to_edge_index(adj_matrix: Tensor) -> Tensor:
    """
    将邻接矩阵转换为PyG格式的edge_index
    Args:
        adj_matrix: 邻接矩阵 [num_nodes, num_nodes]
    Returns:
        edge_index: 边索引 [2, num_edges]
    """
    # 获取非零元素坐标
    rows, cols = torch.where(adj_matrix > 0)

    # 去重处理：只保留i < j的边
    mask = rows < cols
    rows = rows[mask]
    cols = cols[mask]

    # 构造无向图边索引
    edge_index = torch.stack([
        torch.cat([rows, cols]),
        torch.cat([cols, rows])
    ], dim=0)

    return edge_index


def generate_edge_index(
        x: Tensor,
        threshold: float = 0.7,
        top_k: int = None,
        device: str = 'cpu'
) -> Tensor:
    """
    端到端生成edge_index
    Args:
        x: 输入数据 [num_nodes, timesteps]
        threshold: 相关系数阈值
        top_k: 每个节点保留的最大邻居数
        device: 输出设备
    Returns:
        edge_index: [2, num_edges]
    """
    x = x.to(device)

    # 构建邻接矩阵
    adj_matrix = build_adjacency_matrix(
        x,
        threshold=threshold,
        top_k=top_k
    )

    # 转换为edge_index
    edge_index = adjacency_to_edge_index(adj_matrix)

    return edge_index.to(device)

def generate_adjacency_matrix_np(matrix, file_path):
    # 转换为NumPy数组并提取有效点
    points = np.argwhere(np.asarray(matrix) == 1)
    n = points.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.int8)
    # 计算每个点的坐标平方和 (xi² + yi²)
    sum_sq = np.sum(points ** 2, axis=1)
    # 计算点积矩阵 (xi*xj + yi*yj)
    dot_matrix = points @ points.T
    # 通过广播计算所有点对的距离平方 (xi² + yi² + xj² + yj² - 2xi*xj - 2yi*yj)
    dist_sq = sum_sq[:, None] + sum_sq[None, :] - 2 * dot_matrix
    # 生成邻接矩阵 (距离≤2且非对角线)
    adj_matrix = (dist_sq <= 2) & ~np.eye(n, dtype=bool)
    adj_matrix = normalize_adj(adj_matrix)
    # np.savez(file_path, adj=adj_matrix)
    return adj_matrix

def generate_adjacency_matrix_torch(mask):
    # 转换为PyTorch张量并提取有效点
    points = torch.argwhere(mask == 1)
    n = points.shape[0]

    if n == 0:
        adj_matrix = torch.zeros((0, 0), dtype=torch.float32)
    else:
        # 计算每个点的坐标平方和 (xi² + yi²)
        sum_sq = torch.sum(points ** 2, dim=1)

        # 计算点积矩阵 (xi*xj + yi*yj)
        dot_matrix = points @ points.T

        # 计算所有点对的距离平方 (xi² + yi² + xj² + yj² - 2xi*xj - 2yi*yj)
        dist_sq = sum_sq.unsqueeze(1) + sum_sq.unsqueeze(0) - 2 * dot_matrix

        # 生成邻接矩阵 (距离≤2且非对角线)
        adj_matrix = (dist_sq <= 2) & ~torch.eye(n, dtype=torch.bool)
        adj_matrix = adj_matrix.float()

    # 归一化邻接矩阵
    adj_matrix = normalize_adj_TORCH(adj_matrix)

    # 转换为numpy数组并保存
    return adj_matrix


# """用于OVPGCN的距离阈值邻接矩阵生成"""
# def generate_static_adj(valid_indices, mask_list, file_list, file_path, seq_len):
#     for daily, index in enumerate(tqdm(valid_indices, desc='距离阈值邻接矩阵生成')):
#         # gradient_daily = np.where(max_mask, gradient_mmap[daily], np.nan)
#         data_NT = mask_list[index].transpose(0, 1)
#         adj_path = f'{file_path}/{file_list[index]}.pth'
#         adj = generate_adjacency_matrix_np(data_NT, adj_path)
#         torch.save(torch.from_numpy(adj).float(), adj_path)
#         # np.savez(adj_path, adj=adj)


def normalize_adj_TORCH(adj):
    """对称归一化邻接矩阵: D^(-1/2) (A + I) D^(-1/2)"""
    device = adj.device
    n = adj.size(0)
    
    # 1. 添加自环 (避免直接修改原矩阵)
    # adj = adj.clone()  # 避免原地修改
    adj[range(n), range(n)] += 1  # 比 adj + torch.eye(n) 更节省内存
    
    # 2. 计算度矩阵 (避免存储完整的 D^(-1/2) 矩阵)
    row_sum = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    # 3. 对称归一化 (分步计算，避免大矩阵乘法)
    # 先计算 D^(-1/2) @ A
    norm_adj = adj.mul_(d_inv_sqrt.view(-1, 1))  # 广播乘法，比矩阵乘法更高效
    
    # 删除不再需要的变量
    del adj
    torch.cuda.empty_cache()  # 清理未使用的缓存内存
    
    # 使用 in-place 操作继续计算
    norm_adj.mul_(d_inv_sqrt.view(1, -1))
    
    return norm_adj




def normalize_adj(adj):
    """对称归一化邻接矩阵: D^(-1/2) (A + I) D^(-1/2)"""
    # 添加自环 (Add self-loops)
    adj = adj + np.eye(adj.shape[0])

    # 计算度矩阵 (Compute degree matrix)
    row_sum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # 对称归一化 (Symmetric normalization)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return norm_adj


# import numpy as np
# from dtaidistance import dtw
# from dtaidistance.dtw_ndim import distance_matrix
#
# def generate_static_adjacency_matrix(sst_data, file_path):
#     """
#     生成静态邻接矩阵
#     :param sst_data: N*T的numpy数组，N为区域数，T为时间步长
#     :return: N*N的邻接矩阵A_s
#     """
#     sst_data = np.asarray(sst_data, dtype=np.float16, order='C')
#
#     # Step 1: 并行加速的DTW距离矩阵计算
#     dtw_dist = distance_matrix(
#         sst_data.astype(np.float16),  # 使用float32加速计算
#         parallel=True,
#         show_progress=False,
#         compact=False
#     )
#
#     # 归一化与指数变换
#     min_val, max_val = np.min(dtw_dist), np.max(dtw_dist)
#     normalized = (dtw_dist - min_val) / (max_val - min_val + 1e-12)
#     adj = normalize_adj(np.exp(-np.square(normalized)))
#     np.savez(file_path, adj=adj)




if __name__ == "__main__":
    # stats = compute_mmap_statistics(
    #     "/home/jameszhang/桌面/SST/all_data/sst_mmap.npy",
    #     "/home/jameszhang/桌面/SST/sst_features.npy",
    #     safe_chunk_mb=3000 # 根据可用内存调整
    # )
    # print("\n统计结果:")
    # for k, v in stats.items():
    #     print(f"{k:>15}: {v:.4f}")

    data = np.load('/home/jameszhang/桌面/SST/data/all_data/sst_mmap.npy',
                   mmap_mode='r', allow_pickle=True)
    # gradient = compute_instant_gradient(data)
    gradient = np.load('/home/jameszhang/桌面/SST/data/data_gradient/mmap.npy', allow_pickle=True)
    data = data[2:2+30+1]
    gradient = gradient[0]

    valid_mask = ~np.isnan(gradient)
    print(data.shape, len(valid_mask))
    data_flat = data[:, valid_mask]
    print(data_flat.shape)

    hyper_graph_constructor = HyperGraphConstructor(gradient)
    hyper_graph_constructor.classify_sub_graphs()
    sub_graph = hyper_graph_constructor.get_sub_graphs()
    adjusted_daily = data.transpose(1, 2, 0)  # 形状 (H, W, lens+1)
    # 扩展 gradient 维度
    gradient_expanded = gradient[..., np.newaxis]  # 形状 (H, W, 1)
    merged_data = np.concatenate([adjusted_daily, gradient_expanded], axis=2)  # 形状 (H, W, lens+2)

    adj = grid_hypergraph_to_graph(merged_data, np.array(sub_graph, dtype=object), random_state=1)

    print(f"生成的邻接矩阵形状: {adj.shape}") # torch.Size([1083131, 1083131])   596869
    print(f"非零元素数量: {adj._nnz()}") # 非零元素数量: 5386763

    valid_mask = ~np.any(np.isnan(merged_data), axis=2)
    rows, cols = np.where(valid_mask)
    valid_coords = list(zip(rows, cols))  # 有效节点坐标
    adj_nodes = adj.shape[0]  # 邻接矩阵节点数

    assert len(valid_coords) == adj_nodes, "节点数量不一致" # 1680000
    # gradient = calculate_temperature_gradient_time(data[:400])
    # print(gradient.shape)

    # # 使用np.isnan来检测NaN值，然后使用~操作符来取反，将非NaN值的位置设置为True
    # mask = ~np.isnan(arr)
    # # 将布尔矩阵转换为整数矩阵，其中True变为1，False变为0
    # mask_matrix = mask.astype(int)


