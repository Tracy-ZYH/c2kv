import math
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Callable, Dict, Any


class LoRALinear(nn.Linear):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        enable_lora: bool,
        config: Dict[str, Any],
        **kwargs
    ) -> None:
        super().__init__(in_features, out_features, **kwargs)
        self.enable_lora = enable_lora
        if self.enable_lora:
            if (dropout := config["lora_dropout"]) > 0.:
                self.lora_dropout = nn.Dropout(dropout)
            else:
                self.lora_dropout = lambda x: x
            self.rank = config["rank"]
            self.gist_lora_A = nn.Parameter(self.weight.new_zeros((self.rank, in_features)))
            self.gist_lora_B = nn.Parameter(self.weight.new_zeros((out_features, self.rank)))
            self.scaling = config["lora_alpha"] / self.rank
            self.weight.requires_grad = False
            if hasattr(self, 'bias') and self.bias is not None:
                self.bias.requires_grad = False
            self.reset_lora_parameters()
 
    def reset_lora_parameters(self) -> None:
        if self.enable_lora:
            nn.init.kaiming_uniform_(self.gist_lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.gist_lora_B)
    
    def forward_lora(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_lora:
            return torch.tensor(0) # do not change the original output
        x = self.lora_dropout(x) @ self.gist_lora_A.transpose(0, 1)
        return x @ self.gist_lora_B.transpose(0, 1) * self.scaling
    
    def forward(self, x: torch.Tensor, with_gist: bool = False) -> torch.Tensor:
        h = super().forward(x)
        if with_gist and self.enable_lora:
            h = h + self.forward_lora(x)
        return h


def get_linear_cls(config: Dict[str, Any], module_type: str) -> Callable[..., LoRALinear]:
    """Return a factory function to create a LoRALinear module fixed with the given config."""
    enable_lora = module_type in config["lora_modules"]
    def _init_linear(*args, **kwargs):
        return LoRALinear(*args, config=config, enable_lora=enable_lora, **kwargs)
    return _init_linear
