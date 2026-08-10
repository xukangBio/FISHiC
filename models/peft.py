import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        x_d = self.dropout(x)
        delta = F.linear(F.linear(x_d, self.lora_A), self.lora_B) * self.scale
        return base_out + delta


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / float(rank)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        k_h, k_w = base.kernel_size
        in_dim = (base.in_channels // base.groups) * k_h * k_w
        out_dim = base.out_channels

        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        x_d = self.dropout(x)

        delta_w = torch.matmul(self.lora_B, self.lora_A).view(
            self.base.out_channels,
            self.base.in_channels // self.base.groups,
            self.base.kernel_size[0],
            self.base.kernel_size[1],
        )
        delta = F.conv2d(
            x_d,
            delta_w * self.scale,
            bias=None,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
        )
        return base_out + delta


def _set_module_by_name(root: nn.Module, module_name: str, new_module: nn.Module):
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = None,
):
    replaced = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if target_modules is not None and len(target_modules) > 0:
                if not any(k in name for k in target_modules):
                    continue
            if isinstance(m, nn.Conv2d):
                new_m = LoRAConv2d(m, rank=rank, alpha=alpha, dropout=dropout)
            else:
                new_m = LoRALinear(m, rank=rank, alpha=alpha, dropout=dropout)
            _set_module_by_name(model, name, new_m)
            replaced += 1
    return replaced


def mark_only_lora_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True


def count_trainable_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
