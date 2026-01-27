# query_storage.py
import torch
from typing import List, Optional

class QueryStorage:
    """用于管理和存储query tensor的类，支持上下文管理器"""
    def __init__(self, model=None, enabled: bool = False):
        self.model = model
        self.storage: List[torch.Tensor] = []
        self.enabled: bool = enabled
        self.hooks = []
        if model is not None and enabled:
            self.register_hooks(model)
    
    def enable(self):
        """开启存储功能"""
        self.enabled = True
    
    def disable(self):
        """关闭存储功能"""
        self.enabled = False
    
    def clear(self):
        """清除所有存储的tensor"""
        self.storage.clear()

    def get_all_queries(self) -> List[torch.Tensor]:
        """获取所有层的query tensor"""
        return [tensor for tensor in self.storage]
    
    def create_hook(self, layer_idx: int):
        """创建钩子函数"""
        def hook(module, input, output):
            if self.enabled:
                # q_proj输出形状: (batch_size, seq_len, hidden_dim)
                # 确保list足够长
                while len(self.storage) <= layer_idx:
                    self.storage.append(None)
                self.storage[layer_idx] = output.detach()
        return hook
    
    def register_hooks(self, model):
        """在所有层的q_proj上注册钩子"""
        self.model = model
        for idx, layer in enumerate(model.model.layers):
            hook_fn = self.create_hook(idx)
            handle = layer.self_attn.q_proj.register_forward_hook(hook_fn)
            self.hooks.append(handle)
    
    def remove_hooks(self):
        """移除所有已注册的钩子"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def __enter__(self):
        """进入上下文管理器"""
        if self.model is None:
            raise RuntimeError("Model not set. Call register_hooks() first or pass model to __init__")
        self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器时自动unhook"""
        self.disable()
        self.remove_hooks()
        self.clear()
        return False
    
    def __repr__(self):
        return f"QueryStorage(enabled={self.enabled}, num_layers={len(self.storage)}, hooks={len(self.hooks)})"
    
    def __len__(self):
        """返回存储的tensor数量"""
        return len(self.storage)
    
    def __getitem__(self, idx: int):
        """支持索引访问"""
        return self.storage[idx]
