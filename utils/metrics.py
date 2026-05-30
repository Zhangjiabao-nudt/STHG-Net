# import torch
from typing import Optional


# def accuracy(pred, y):
#     """
#     :param pred: predictions
#     :param y: ground truth
#     :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
#     """
#     return 1 - torch.lin0alg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


# def grid_accuracy(pred, y):
#     """
#     计算四维张量的准确率，输入形状为 (B, t, H, W)

#     :param pred: 预测值，形状 (B, t, H, W)
#     :param y: 真实值，形状 (B, t, H, W)
#     :return: 准确率，定义 1 - (范数(y - pred) / 范数(y))
#     """
#     # 计算每个样本每个时间步的范数
#     dims = (2, 3)  # 在 H, W 维度计算范数

#     # 计算差异范数 (B, t)
#     norm_diff = torch.linalg.norm(y - pred, ord="fro", dim=dims)

#     # 计算真实值范数 (B, t)
#     norm_y = torch.linalg.norm(y, ord="fro", dim=dims)

#     # 避免除以零 (将零替换为1e-6)
#     norm_y = torch.where(norm_y == 0, 1e-6, norm_y)

#     # 计算每个样本每个时间步的准确率 (B, t)
#     acc_per_step = 1 - (norm_diff / norm_y)

#     # 返回整体平均值（可根据需求修改聚合方式）
#     return acc_per_step.mean()  # 标量输出

# def r2(pred, y):
#     """
#     :param y: ground truth
#     :param pred: predictions
#     :return: R square (coefficient of determination)
#     """
#     return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


# def explained_variance(pred, y):
#     return 1 - torch.var(y - pred) / torch.var(y)


# def mean_absolute_error(y_true, y_pred):
#     """
#     计算两个张量之间的平均绝对误差（MAE）

#     参数：
#     y_true -- 真实值张量（支持任意维度）
#     y_pred -- 预测值张量（需与y_true形状相同）

#     返回：
#     mae -- 标量张量（保留梯度信息）

#     特性：
#     - 自动微分兼容
#     - GPU加速支持
#     - O(n)时间复杂度

#     示例：
#     >>> y_true = torch.tensor([3.0, -0.5, 2.0], dtype=torch.float32)
#     >>> y_pred = torch.tensor([2.5, 0.0, 2.5], dtype=torch.float32)
#     >>> mae = mean_absolute_error(y_true, y_pred)
#     >>> print(f"{mae.item():.2f}")
#     0.67
#     """
#     # 形状一致性验证
#     if y_true.shape != y_pred.shape:
#         raise ValueError(f"形状不匹配: y_true {tuple(y_true.shape)}, y_pred {tuple(y_pred.shape)}")

#     # 向量化计算流程
#     absolute_errors = torch.abs(y_true - y_pred)
#     return torch.mean(absolute_errors)


# def root_mean_squared_error(y_true, y_pred):
#     """
#     计算两个张量之间的均方根误差（RMSE）

#     参数：
#     y_true -- 真实值张量（支持任意维度）
#     y_pred -- 预测值张量（需与y_true形状相同）

#     返回：
#     rmse -- 标量张量（保留梯度信息）

#     示例：
#     >>> y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
#     >>> y_pred = torch.tensor([2.5, 0.0, 2.0, 8.0])
#     >>> rmse = root_mean_squared_error(y_true, y_pred)
#     >>> print(f"{rmse.item():.3f}")
#     0.612
#     """
#     # 形状一致性验证
#     if y_true.shape != y_pred.shape:
#         raise ValueError(f"形状不匹配: y_true {tuple(y_true.shape)}, y_pred {tuple(y_pred.shape)}")

#     # 向量化计算流程（保持梯度）
#     squared_errors = torch.square(y_true - y_pred)  # 自动广播机制
#     mean_squared = torch.mean(squared_errors)  # 全局平均
#     return torch.sqrt(mean_squared)  # 稳定开平方


import torch


def accuracy(pred, y, mask=None):
    """
    带掩码支持的准确率计算，定义 1 - (差异范数 / 真实值范数)
    """
    if mask is not None:
        # assert mask.shape == y.shape, "Mask shape must match input shape"
        # mask = mask.float()
        diff = (y - pred) * mask
        norm_diff = torch.linalg.norm(diff, "fro")
        norm_y = torch.linalg.norm(y * mask, "fro")
        norm_y = torch.where(norm_y == 0, 1e-6, norm_y)
        return 1 - (norm_diff / norm_y)
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def grid_accuracy(pred, y, mask=None):
    """
    带掩码支持的四维张量准确率计算 (B, t, H, W)
    """
    dims = (2, 3)  # H, W 维度

    if mask is not None:
        # assert mask.shape == y.shape, "Mask shape must match input shape"
        # mask = mask.float()

        # 计算掩码区域范数
        squared_diff = (y - pred) ** 2 * mask
        norm_diff = torch.sqrt(torch.sum(squared_diff, dim=dims))

        squared_y = y ** 2 * mask
        norm_y = torch.sqrt(torch.sum(squared_y, dim=dims))

        norm_y = torch.where(norm_y == 0, 1e-6, norm_y)
        acc_per_step = 1 - (norm_diff / norm_y)
    else:
        norm_diff = torch.linalg.norm(y - pred, ord="fro", dim=dims)
        norm_y = torch.linalg.norm(y, ord="fro", dim=dims)
        norm_y = torch.where(norm_y == 0, 1e-6, norm_y)
        acc_per_step = 1 - (norm_diff / norm_y)

    return acc_per_step.mean()


def r2(pred, y, mask=None):
    """
    带掩码支持的决定系数 (R²)
    """
    if mask is not None:
        # assert mask.shape == y.shape, "Mask shape must match input shape"
        # mask = mask.float()

        # 计算残差平方和
        ss_res = torch.sum(((y - pred) ** 2) * mask)

        # 计算总平方和
        y_masked = y * mask
        y_mean = torch.sum(y_masked) / torch.sum(mask).clamp(min=1e-8)
        ss_tot = torch.sum(((y_masked - y_mean) ** 2))
    else:
        ss_res = torch.sum((y - pred) ** 2)
        y_mean = torch.mean(y)
        ss_tot = torch.sum((y - y_mean) ** 2)

    return 1 - ss_res / ss_tot.clamp(min=1e-8)


def explained_variance(pred, y, mask=None):
    """
    带掩码支持的解释方差
    """
    if mask is not None:
        # assert mask.shape == y.shape, "Mask shape must match input shape"
        # mask = mask.float()

        # 残差计算
        res = (y - pred) * mask
        mean_res = torch.sum(res) / torch.sum(mask).clamp(min=1e-8)
        var_res = torch.sum((res - mean_res) ** 2) / torch.sum(mask).clamp(min=1e-8)

        # 真实值方差
        y_masked = y * mask
        mean_y = torch.sum(y_masked) / torch.sum(mask).clamp(min=1e-8)
        var_y = torch.sum((y_masked - mean_y) ** 2) / torch.sum(mask).clamp(min=1e-8)
    else:
        var_res = torch.var(y - pred, unbiased=False)
        var_y = torch.var(y, unbiased=False)

    return 1 - (var_res / var_y.clamp(min=1e-8))


# def mean_absolute_error(y_true, y_pred, mask=None):
#     """
#     带掩码支持的平均绝对误差 (MAE)
#     """
#     if y_true.shape != y_pred.shape:
#         raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
#
#     abs_errors = torch.abs(y_true - y_pred)
#
#     if mask is not None:
#         # assert mask.shape == y_true.shape, "Mask shape must match input shape"
#         # mask = mask.float()
#         return torch.sum(abs_errors * mask) / torch.sum(mask).clamp(min=1e-8)
#     return torch.mean(abs_errors)
#
#
# def root_mean_squared_error(y_true, y_pred, mask=None):
#     """
#     带掩码支持的均方根误差 (RMSE)
#     """
#     if y_true.shape != y_pred.shape:
#         raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
#
#     squared_errors = (y_true - y_pred) ** 2
#
#     if mask is not None:
#         # assert mask.shape == y_true.shape, "Mask shape must match input shape"
#         # mask = mask.float()
#         mean_squared = torch.sum(squared_errors * mask) / torch.sum(mask).clamp(min=1e-8)
#     else:
#         mean_squared = torch.mean(squared_errors)
#
#     return torch.sqrt(mean_squared)
#
#
# def root_mean_position_error(y_true, y_pred, mask=None):
#     """
#     带掩码支持的位置均方根误差 (RMPE)
#     """
#     if y_true.shape != y_pred.shape:
#         raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
#
#     # 计算平方误差
#     squared_errors = torch.square(y_true - y_pred)
#
#     # 处理掩码逻辑
#     if mask is not None:
#         # assert mask.shape == y_true.shape, "Mask shape must match input shape"
#         # mask = mask.float()
#
#         # 计算有效区域的平方误差总和
#         sum_squared = torch.sum(squared_errors * mask)
#
#         # 计算有效元素数量（避免除以零）
#         valid_count = torch.sum(mask).clamp(min=1e-8)
#
#         # 计算均方根误差
#         return torch.sqrt(sum_squared / valid_count)
#
#     # 无掩码情况下的计算
#     return torch.sqrt(torch.mean(squared_errors))
#
#
# def masked_mape(y_true: torch.Tensor,
#                 y_pred: torch.Tensor,
#                 mask: Optional[torch.Tensor] = None,
#                 epsilon: float = None) -> torch.Tensor:
#
#
#     # 自动设置最小epsilon值（根据张量数据类型）
#     if epsilon is None:
#         epsilon = torch.finfo(y_true.dtype).eps
#
#     # 计算绝对百分比误差
#     # abs_per_error = torch.abs((y_true - y_pred) / (y_true + epsilon))  # [..., D]
#     abs_per_error = torch.abs((y_true - y_pred) / (y_true+epsilon))
#
#     # 掩码处理
#     if mask is not None:
#         # print(mask[0,:])
#         # 计算有效区域的平均误差
#
#         valid_errors = torch.sum(abs_per_error * mask)
#         if valid_errors.numel() == 0:  # 避免空掩码
#             return torch.tensor(float('nan'), device=y_true.device)
#         valid_count = torch.sum(mask)
#         return valid_errors / valid_count
#
#     # 无掩码情况
#     return torch.mean(abs_per_error)


import torch
from typing import Optional

import torch
from typing import Optional


def mean_absolute_error_time(y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    带掩码支持的平均绝对误差 (MAE)，保留时间维度

    参数:
        y_true: 真实值，形状 (Batch, Time, Height, Width)
        y_pred: 预测值，形状 (Batch, Time, Height, Width)
        mask:   掩码，形状 (Batch, Height, Width) 或 None

    返回:
        MAE 张量，形状 [Time, 1]
    """
    # 检查输入维度
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    if y_true.dim() != 4 or y_pred.dim() != 4:
        raise ValueError(f"Inputs must be 4D tensors (B, T, H, W), got y_true {y_true.dim()}D, y_pred {y_pred.dim()}D")

    # 计算绝对误差 (B, T, H, W)
    abs_errors = torch.abs(y_true - y_pred)

    # 处理掩码逻辑
    if mask is not None:
        # 检查掩码维度
        if mask.dim() != 3 or mask.shape != y_true.shape[:1] + y_true.shape[2:]:
            raise ValueError(f"Mask shape {mask.shape} must be (B, H, W)")

        # 扩展掩码维度以匹配输入 (B, 1, H, W)
        mask_expanded = mask.unsqueeze(1)  # 添加时间维度

        # 应用掩码并在(B, H, W)维度上求和
        sum_abs = torch.sum(abs_errors * mask_expanded, dim=[0, 2, 3])  # 结果形状 [T]
        # 计算每个时间步的有效像素数
        valid_count = torch.sum(mask_expanded, dim=[0, 2, 3]).clamp(min=1e-8)  # 结果形状 [T]

        # 计算每个时间步的MAE并调整形状为[T, 1]
        return (sum_abs / valid_count).unsqueeze(1)

    # 无掩码情况：在(B, H, W)维度上求平均，保留时间维度
    return torch.mean(abs_errors, dim=[0, 2, 3]).unsqueeze(1)  # 形状 [T, 1]


def root_mean_squared_error_time(y_true: torch.Tensor,
                            y_pred: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    带掩码支持的均方根误差 (RMSE)，保留时间维度

    参数:
        y_true: 真实值，形状 (B, T, H, W)
        y_pred: 预测值，形状 (B, T, H, W)
        mask:   掩码，形状 (B, H, W) 或 None

    返回:
        RMSE 张量，形状 [T, 1]
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    # 计算平方误差 (B, T, H, W)
    squared_errors = (y_true - y_pred) ** 2

    if mask is not None:
        if mask.dim() != 3 or mask.shape != y_true.shape[0:1] + y_true.shape[2:4]:
            raise ValueError(f"Mask shape {mask.shape} must be (B, H, W)")

        # 扩展mask到 (B, 1, H, W) 以便广播
        mask_expanded = mask.unsqueeze(1)  # (B, 1, H, W)

        # 应用掩码并在(B, H, W)维度上求和
        sum_squared = torch.sum(squared_errors * mask_expanded, dim=[0, 2, 3])  # 结果形状 [T]
        # 计算每个时间步的有效像素数
        valid_count = torch.sum(mask_expanded, dim=[0, 2, 3]).clamp(min=1e-8)  # 结果形状 [T]

        # 计算每个时间步的MSE
        mean_squared = sum_squared / valid_count  # 形状 [T]
    else:
        # 在(B, H, W)维度上求平均，保留时间维度
        mean_squared = torch.mean(squared_errors, dim=[0, 2, 3])  # 形状 [T]

    # 计算RMSE并调整形状为[T, 1]
    return torch.sqrt(mean_squared).unsqueeze(1)



def mean_absolute_error(y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    带掩码支持的平均绝对误差 (MAE)

    参数:
        y_true: 真实值，形状 (Batch, Time, Height, Width)
        y_pred: 预测值，形状 (Batch, Time, Height, Width)
        mask:   掩码，形状 (Batch, Height, Width) 或 None

    返回:
        MAE 标量值

    示例:
        >>> B, T, H, W = 4, 10, 64, 64
        >>> y_true = torch.rand(B, T, H, W)
        >>> y_pred = torch.rand(B, T, H, W)
        >>> mask = torch.randint(0, 2, (B, H, W)).float()
        >>> mae = mean_absolute_error(y_true, y_pred, mask)
    """
    # 检查输入维度
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    if y_true.dim() != 4 or y_pred.dim() != 4:
        raise ValueError(f"Inputs must be 4D tensors (B, T, H, W), got y_true {y_true.dim()}D, y_pred {y_pred.dim()}D")

    # 计算绝对误差 (B, T, H, W)
    abs_errors = torch.abs(y_true - y_pred)

    # 处理掩码逻辑
    if mask is not None:
        # 检查掩码维度
        if mask.dim() != 3 or mask.shape != y_true.shape[:1] + y_true.shape[2:]:
            raise ValueError(f"Mask shape {mask.shape} must be (B, H, W)")

        # 扩展掩码维度以匹配输入 (B, 1, H, W)
        mask_expanded = mask.unsqueeze(1)  # 添加时间维度

        # 计算有效区域的MAE
        sum_abs = torch.sum(abs_errors * mask_expanded)
        T = y_true.size(1)
        valid_count = T * torch.sum(mask_expanded).clamp(min=1e-8)
        return sum_abs / valid_count

    # 无掩码情况
    return torch.mean(abs_errors)


def root_mean_squared_error(y_true: torch.Tensor,
                            y_pred: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    带掩码支持的均方根误差 (RMSE)
    输入:
        y_true: 真实值，形状 (B, T, H, W)
        y_pred: 预测值，形状 (B, T, H, W)
        mask:   掩码，形状 (B, H, W) 或 None
    返回:
        RMSE 标量值
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    # 计算平方误差 (B, T, H, W)
    squared_errors = (y_true - y_pred) ** 2

    if mask is not None:
        if mask.dim() != 3 or mask.shape != y_true.shape[0:1] + y_true.shape[2:4]:
            raise ValueError(f"Mask shape {mask.shape} must be (B, H, W)")

        # 扩展mask到 (B, 1, H, W) 以便广播
        mask_expanded = mask.unsqueeze(1)  # (B, 1, H, W)

        # 计算有效区域的均方误差
        sum_squared = torch.sum(squared_errors * mask_expanded)
        T = y_true.size(1)
        valid_count = T*torch.sum(mask_expanded).clamp(min=1e-8)
        mean_squared = sum_squared / valid_count
    else:
        mean_squared = torch.mean(squared_errors)

    return torch.sqrt(mean_squared)


def root_mean_position_error(y_true: torch.Tensor,
                             y_pred: torch.Tensor,
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    带掩码支持的位置均方根误差 (RMPE)
    输入输出维度要求同 RMSE 函数
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    # 计算平方误差 (B, T, H, W)
    squared_errors = (y_true - y_pred) ** 2

    if mask is not None:
        if mask.dim() != 3 or mask.shape != y_true.shape[0:1] + y_true.shape[2:4]:
            raise ValueError(f"Mask shape {mask.shape} must be (B, H, W)")

        # 扩展mask到 (B, 1, H, W)
        mask_expanded = mask.unsqueeze(1)

        # 计算有效区域的误差
        sum_squared = torch.sum(squared_errors * mask_expanded)
        T = y_true.size(1)
        valid_count = T * torch.sum(mask_expanded).clamp(min=1e-8)
        mean_squared = sum_squared / valid_count
    else:
        mean_squared = torch.mean(squared_errors)

    return torch.sqrt(mean_squared)


def masked_mape(y_true: torch.Tensor,
                       y_pred: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       epsilon: float = None) -> torch.Tensor:
    """
    带掩码支持的平均绝对百分比误差 (MAPE)
    输入温度自动转换为开尔文避免零值问题
    输入:
        y_true: 真实值（摄氏度），形状 (B, T, H, W)
        y_pred: 预测值（摄氏度），形状 (B, T, H, W)
        mask:   掩码，形状 (B, H, W) 或 None
        epsilon: 保留参数（已弃用，保持接口兼容）
    返回:
        MAPE百分比值
    """
    # 转换为开尔文温度
    y_true_k = y_true + 273.15
    y_pred_k = y_pred + 273.15

    # 计算绝对百分比误差 (B, T, H, W)
    abs_per_error = torch.abs((y_true_k - y_pred_k) / y_true_k)

    if mask is not None:
        if mask.dim() != 3 or mask.shape != y_true.shape[0:1] + y_true.shape[2:4]:
            raise ValueError(f"Mask shape {mask.shape} must be (B, H, W)")

        # 扩展mask到 (B, 1, H, W)
        mask_expanded = mask.unsqueeze(1)

        # 计算有效区域误差
        valid_errors = torch.sum(abs_per_error * mask_expanded)
        T = y_true.size(1)
        valid_count = T * torch.sum(mask_expanded).clamp(min=1e-8)

        if valid_count == 0:
            return torch.tensor(float('nan'), device=y_true.device)

        return (valid_errors / valid_count) * 100  # 修正百分比转换

    else:
        return torch.mean(abs_per_error) * 100

def masked_mape(y_true: torch.Tensor,
                       y_pred: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       epsilon: float = None) -> torch.Tensor:
    """
    带掩码支持的平均绝对百分比误差 (MAPE)
    输入温度自动转换为开尔文避免零值问题
    输入:
        y_true: 真实值（摄氏度），形状 (B, T, H, W)
        y_pred: 预测值（摄氏度），形状 (B, T, H, W)
        mask:   掩码，形状 (B, H, W) 或 None
        epsilon: 保留参数（已弃用，保持接口兼容）
    返回:
        MAPE百分比值
    """
    # 转换为开尔文温度
    y_true_k = y_true + 273.15
    y_pred_k = y_pred + 273.15

    # 计算绝对百分比误差 (B, T, H, W)
    abs_per_error = torch.abs((y_true_k - y_pred_k) / y_true_k)

    if mask is not None:
        if mask.dim() != 3 or mask.shape != y_true.shape[0:1] + y_true.shape[2:4]:
            raise ValueError(f"Mask shape {mask.shape} must be (B, H, W)")

        # 扩展mask到 (B, 1, H, W)
        mask_expanded = mask.unsqueeze(1)

        # 计算有效区域误差
        valid_errors = torch.sum(abs_per_error * mask_expanded)
        T = y_true.size(1)
        valid_count = T * torch.sum(mask_expanded).clamp(min=1e-8)

        if valid_count == 0:
            return torch.tensor(float('nan'), device=y_true.device)

        return (valid_errors / valid_count) * 100  # 修正百分比转换

    else:
        return torch.mean(abs_per_error) * 100


if __name__ == '__main__':
    B, T, H, W = 4, 10, 64, 64
    y_true = torch.randn(B, T, H, W)
    y_pred = torch.randn(B, T, H, W)
    mask = torch.randint(0,  2, (B, H, W)).float()  # 随机二值掩码

    rmse = root_mean_squared_error(y_true, y_pred, mask)
    rmpe = root_mean_position_error(y_true, y_pred, mask)
    mape = masked_mape(y_true, y_pred, mask)

    print(f"RMSE: {rmse.item():.4f}, RMPE: {rmpe.item():.4f}, MAPE: {mape.item():.4f}")
    B, T, H, W = 4, 10, 64, 64
    y_true = torch.rand(B, T, H, W)
    y_pred = torch.rand(B, T, H, W)
    mask = torch.randint(0, 2, (B, H, W)).float()  # 随机二值掩码

    mae = mean_absolute_error(y_true, y_pred, mask)
    print(f"MAE: {mae.item():.4f}")