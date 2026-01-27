import math
import torch
import torch.nn.functional as F
from typing import Tuple


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(num_key_value_heads * n_rep, slen, head_dim)


def compress_kv_snapkv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    window_size: int = 64,
    max_capacity_prompt: int = 320,
    kernel_size: int = 5,
    pooling: str = 'avgpool',
    num_heads: int = None,
    num_kv_heads: int = None,
    head_dim: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    根据attention score压缩KV缓存 (支持GQA)
    
    Args:
        query_states: (seq_len, hidden_size)
        key_states: (num_key_value_heads, seq_len, head_dim)
        value_states: (num_key_value_heads, seq_len, head_dim)
        window_size: 滑动窗口大小，保留最近的token
        max_capacity_prompt: 最大KV容量
        kernel_size: 池化核大小
        pooling: 池化方法 ('avgpool' 或 'maxpool')
        num_heads: 注意头数，如果为None则从query推断
        num_kv_heads: KV头数，如果为None则从key_states推断
        head_dim: 头维度，如果为None则从key_states推断
    
    Returns:
        compressed_key_states: (num_key_value_heads, max_capacity_prompt, head_dim)
        compressed_value_states: (num_key_value_heads, max_capacity_prompt, head_dim)
        indices: (num_kv_heads, num_compress)
    """
    seq_len, hidden_size = query_states.shape
    
    # 从输入推断参数
    if num_kv_heads is None:
        num_kv_heads = key_states.shape[0]
    if head_dim is None:
        head_dim = key_states.shape[-1]
    if num_heads is None:
        num_heads = hidden_size // head_dim
    
    # 计算GQA的重复系数
    n_rep = num_heads // num_kv_heads
    
    window_indices = torch.arange(seq_len, device=query_states.device)
    # 如果序列长度未超过容量限制，直接返回
    if seq_len < max_capacity_prompt:
        return key_states, value_states, window_indices

    windows_indices = window_indices[-window_size:]
    if max_capacity_prompt <= window_size:
        key_states, value_states = key_states[:, -max_capacity_prompt:, :], value_states[:, -max_capacity_prompt:, :]
        return key_states, value_states, windows_indices[-max_capacity_prompt:]
    
    # 将query_states从(seq_len, hidden_size)转换为(num_heads, seq_len, head_dim)
    query_states_reshaped = query_states.reshape(seq_len, num_heads, head_dim)
    query_states_reshaped = query_states_reshaped.transpose(0, 1)  # (num_heads, seq_len, head_dim)
    
    # 为了计算attention，需要将KV扩展到num_heads
    k_expanded = repeat_kv(key_states, n_rep)  # (num_heads, seq_len, head_dim)
    
    # 步骤1: 计算attention weights
    # 只使用最近window_size个query计算attention
    q_window = query_states_reshaped[:, -window_size:, :]  # (num_heads, window_size, head_dim)
    
    attn_weights = torch.matmul(
        q_window,  # (num_heads, window_size, head_dim)
        k_expanded.transpose(1, 2)  # (num_heads, head_dim, seq_len)
    ) / math.sqrt(head_dim)
    # attn_weights: (num_heads, window_size, seq_len)
    
    # 步骤2: 创建因果mask（只关注当前和过去的token）
    mask = torch.full(
        (window_size, window_size),
        torch.finfo(attn_weights.dtype).min,
        device=attn_weights.device
    )
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    
    # 应用mask到最近window_size个token的attention
    attn_weights[:, -window_size:, -window_size:] += mask[None, :, :]
    
    # 步骤3: softmax归一化
    attn_weights = torch.softmax(attn_weights, dim=-1)
    
    # 步骤4: 聚合过去token的attention score
    # 计算对过去token的关注度（不包括最近window_size个token）
    attn_weights_sum = attn_weights[:, -window_size:, :-window_size].sum(dim=-2)
    # attn_weights_sum: (num_heads, seq_len - window_size)
    
    # 步骤5: 使用pooling平滑attention score
    if pooling == 'avgpool':
        attn_cache = F.avg_pool1d(
            attn_weights_sum.unsqueeze(0),  # (1, num_heads, seq_len - window_size)
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1
        ).squeeze(0)  # (num_heads, seq_len - window_size)
    elif pooling == 'maxpool':
        attn_cache = F.max_pool1d(
            attn_weights_sum.unsqueeze(0),  # (1, num_heads, seq_len - window_size)
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1
        ).squeeze(0)  # (num_heads, seq_len - window_size)
    else:
        raise ValueError(f'Pooling method "{pooling}" not supported. Use "avgpool" or "maxpool"')
    
    # 步骤6: 对num_kv_heads进行平均聚合（因为多个attention head共享同一组KV）
    # 将attention weights从num_heads平均池化回num_kv_heads
    attn_cache_kv = attn_cache.reshape(num_kv_heads, n_rep, -1).mean(dim=1)
    # attn_cache_kv: (num_kv_heads, seq_len - window_size)
    
    # 步骤7: 选择top-k最重要的token
    num_compress = max_capacity_prompt - window_size
    indices = attn_cache_kv.topk(num_compress, dim=-1).indices.sort().values
    # indices: (num_kv_heads, num_compress)
    
    # 步骤8: 扩展indices用于gather操作
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, head_dim)
    # indices_expanded: (num_kv_heads, num_compress, head_dim)
    
    # 步骤9: 选择压缩后的KV
    k_past = key_states[:, :-window_size, :]  # (num_kv_heads, seq_len - window_size, head_dim)
    v_past = value_states[:, :-window_size, :]  # (num_kv_heads, seq_len - window_size, head_dim)
    
    k_past_compress = k_past.gather(dim=1, index=indices_expanded)
    v_past_compress = v_past.gather(dim=1, index=indices_expanded)
    # k_past_compress, v_past_compress: (num_kv_heads, num_compress, head_dim)
    
    # 步骤10: 拼接压缩的过去token和最近的window_size个token
    k_cur = key_states[:, -window_size:, :]  # (num_kv_heads, window_size, head_dim)
    v_cur = value_states[:, -window_size:, :]  # (num_kv_heads, window_size, head_dim)
    
    compressed_key_states = torch.cat([k_past_compress, k_cur], dim=1)
    compressed_value_states = torch.cat([v_past_compress, v_cur], dim=1)
    
    return compressed_key_states, compressed_value_states, torch.cat([indices[0], windows_indices])


def compress_kv(
    method: str, capacity: int, 
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """    压缩KV缓存
    
    Args:
        method: 压缩方法
        compress_rate: 压缩率
        query, key, value: 键值
        **kwargs: 压缩方法所需的参数
    
    Returns:
        压缩后的键值和索引
    """
    if method == 'snapkv':
        kwargs['max_capacity_prompt'] = capacity
        return compress_kv_snapkv(query, key, value, **kwargs)
    raise ValueError(f'Compression method "{method}" not supported.')
