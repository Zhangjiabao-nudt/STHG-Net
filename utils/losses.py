import torch


def mse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3, mask=None):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    mse_loss = masked_mse(targets, inputs, mask)
    print("reg_loss", reg_loss.item())
    print("mse_loss", mse_loss.item())
    return mse_loss + reg_loss


# def weighted_mse(y_true, y_pred, sample_weight=None):
#     squared_diffs = (y_true - y_pred) ** 2
#     if sample_weight is not None:
#         assert sample_weight.shape == y_true.shape[0], "Weight shape mismatch"
#         return (squared_diffs * sample_weight.view(-1, 1)).mean()
#     return squared_diffs.mean()

import torch
from typing import Optional

import torch
from typing import Optional

def masked_mse(y_true: torch.Tensor,
               y_pred: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    带掩码支持的均方误差 (MSE)

    参数:
        y_true: 真实值，形状 (Batch, Time, Height, Width)
        y_pred: 预测值，形状 (Batch, Time, Height, Width)
        mask:   掩码，形状 (Batch, Height, Width) 或 None

    返回:
        MSE 标量值

    示例:
        >>> B, T, H, W = 4, 10, 64, 64
        >>> y_true = torch.rand(B, T, H, W)
        >>> y_pred = torch.rand(B, T, H, W)
        >>> mask = torch.randint(0, 2, (B, H, W)).float()
        >>> mse = masked_mse(y_true, y_pred, mask)
    """
    # 检查输入维度
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    if y_true.dim() != 4 or y_pred.dim() != 4:
        raise ValueError(f"Inputs must be 4D tensors (B, T, H, W), got y_true {y_true.dim()}D, y_pred {y_pred.dim()}D")

    # 计算平方差
    squared_diffs = (y_true - y_pred) ** 2  # (B, T, H, W)

    if mask is not None:
        # 检查掩码维度
        if mask.dim() != 3 or mask.shape != y_true.shape[:1] + y_true.shape[2:]:
            raise ValueError(f"Mask shape {mask.shape} must be (B, H, W)")

        # 扩展掩码维度以匹配输入 (B, 1, H, W)
        mask_expanded = mask.unsqueeze(1).float()  # 转换为浮点类型

        # 应用掩码
        masked_squared = squared_diffs * mask_expanded

        # 计算有效元素数量，避免除零
        T = y_true.size(1)
        valid_elements = (mask.sum() * T).clamp(min=1e-8)

        # 返回掩码区域的均方误差
        return masked_squared.sum() / valid_elements
    else:
        # 无掩码时返回标准MSE
        return torch.mean(squared_diffs)

def _process_mask(mask: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """统一处理不同维度的掩码，返回广播后的浮点型掩码"""
    assert mask.dim() in [2, 4], "Mask must be 2D (H,W) or 4D (B,T,H,W)"

    # 2D掩码处理：广播到与输入张量相同维度
    if mask.dim() == 2:
        H, W = mask.shape
        mask = mask.view(1, 1, H, W)  # 增加B,T维度
        mask = mask.expand_as(ref_tensor)  # 广播到(B,T,H,W)

    # 维度一致性验证
    assert mask.shape == ref_tensor.shape, f"Mask shape {mask.shape} mismatch with input {ref_tensor.shape}"