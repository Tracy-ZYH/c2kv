import torch
import math
from typing import Tuple, List

def rotate_k_cache_rope(
    k_cache: torch.Tensor,
    delta_pos: int,
    rope_theta: float,
) -> torch.Tensor:
    """
    将KV Cache中的RoPE位置编码旋转到新的起始位置
    
    参数:
        k_cache: 原始Key Cache，格式为(heads, seq_len, head_size)
        delta_pos: 位置改变量
    
    返回:
        旋转后的KV Cache，格式与输入相同
    """
    if delta_pos == 0:
        return kv_cache
        
    # 计算位置偏移量
    num_heads, seq_len, head_size = k_cache.shape
    device = k_cache.device
    dtype = k_cache.dtype
    
    # 计算旋转角度（Llama风格的RoPE）
    theta = 1.0 / (rope_theta ** (torch.arange(0, head_size, 2, dtype=dtype, device=device) / head_size))
    
    # 计算旋转角度
    delta = theta * delta_pos
    
    delta = torch.cat((delta, delta), dim=-1)
    delta.view((1, 1, head_size))

    cos_vals = delta.cos()
    sin_vals = delta.sin()
    
    # 应用旋转操作
    k_rotated = k_cache * cos_vals + _rotate_half(k_cache) * sin_vals
    
    return k_rotated

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将输入张量的后半部分旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
    # shape = x.shape
    # x1, x2 = x[..., ::2], x[..., 1::2]
    # return torch.stack([-x2, x1], dim=-1).view(shape)

def rotate_yarn_position_encoding(
    key_tensor: torch.Tensor,
    old_start_pos: int,
    new_start_pos: int,
    rope_theta: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_seq_len: int = 4096,
    extrapolation_factor: float = 1.0
) -> torch.Tensor:
    """
    将YARN位置编码旋转到新的起始位置
    
    参数:
        key_tensor: 输入key tensor，形状为(batch_size, num_head, seq_len, head_size)
        old_start_pos: 原始序列的起始位置
        new_start_pos: 要旋转到的新起始位置
        rope_theta: RoPE的基础频率参数，默认为10000.0
        scaling_factor: YARN缩放因子，用于调整频率
        original_max_seq_len: 原始训练的最大序列长度
        extrapolation_factor: 外推因子，用于控制外推行为
    
    返回:
        旋转后的key tensor，形状与输入相同
    """
    if old_start_pos == new_start_pos:
        return key_tensor
    
    num_heads, seq_len, head_size = key_tensor.shape
    device = key_tensor.device
    dtype = key_tensor.dtype
    
    # YARN特有的频率缩放计算
    if scaling_factor != 1.0:
        # YARN的频率缩放公式
        scale = scaling_factor
        low_freq_factor = math.log(scale) / math.log(original_max_seq_len)
        high_freq_factor = (math.log(scale) - math.log(original_max_seq_len)) / (
            math.log(rope_theta) - math.log(original_max_seq_len))
    else:
        low_freq_factor = 1.0
        high_freq_factor = 1.0
    
    # 计算YARN调整后的theta
    dim_indices = torch.arange(0, head_size, 2, dtype=dtype, device=device)
    # YARN的频率调整：低频维度缩放较少，高频维度缩放较多
    adjusted_theta = rope_theta ** (
        -dim_indices / head_size * (1 - high_freq_factor) * extrapolation_factor
    )
    
    # 创建位置索引
    positions = torch.arange(seq_len, dtype=dtype, device=device) + old_start_pos
    new_positions = torch.arange(seq_len, dtype=dtype, device=device) + new_start_pos
    
    # 应用YARN的位置缩放
    if scaling_factor != 1.0:
        positions = positions * low_freq_factor
        new_positions = new_positions * low_freq_factor
    
    # 计算旋转角度
    freqs = torch.outer(positions, adjusted_theta)
    new_freqs = torch.outer(new_positions, adjusted_theta)
    
    # 计算旋转矩阵的cos和sin分量
    cos_old = torch.cos(freqs)
    sin_old = torch.sin(freqs)
    cos_new = torch.cos(new_freqs)
    sin_new = torch.sin(new_freqs)
    
    # 计算相对旋转矩阵
    cos_delta = cos_new * cos_old + sin_new * sin_old  # cos(Δθ) = cos(θ2-θ1)
    sin_delta = sin_new * cos_old - cos_new * sin_old  # sin(Δθ) = sin(θ2-θ1)
    
    # 扩展维度以便广播 (seq_len, head_size/2) -> (1, 1, seq_len, head_size/2)
    cos_delta = cos_delta.view(1, seq_len, head_size // 2)
    sin_delta = sin_delta.view(1, seq_len, head_size // 2)
    
    # 将cos和sin值重复以匹配head_size
    cos_delta = cos_delta.repeat_interleave(2, dim=-1)
    sin_delta = sin_delta.repeat_interleave(2, dim=-1)
    
    # 应用旋转操作
    rotated_key = key_tensor * cos_delta + _yarn_rotate_half(key_tensor) * sin_delta
    
    return rotated_key

def _yarn_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    YARN风格的半旋转操作
    对向量的每两个分量进行旋转
    """
    x1 = x[..., ::2]  # 偶数索引分量
    x2 = x[..., 1::2]  # 奇数索引分量
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

def yarn_interpolation_factor(
    current_length: int,
    original_max_length: int,
    scaling_factor: float = 1.0
) -> float:
    """
    计算YARN的插值因子
    
    参数:
        current_length: 当前序列长度
        original_max_length: 原始训练的最大长度
        scaling_factor: 缩放因子
    
    返回:
        插值因子，用于动态调整旋转
    """
    if current_length <= original_max_length:
        return 1.0
    
    # YARN的动态插值公式
    ratio = current_length / original_max_length
    return math.sqrt(1 + math.log(ratio) / math.log(original_max_length)) * scaling_factor
