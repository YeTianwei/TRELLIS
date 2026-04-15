from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.sparse import SparseTensor
from ..modules.sparse.linear import SparseLinear


@dataclass
class LoraSpec:
    target_suffixes: Tuple[str, ...]
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, (nn.Linear, SparseLinear)):
            raise TypeError(f"Unsupported LoRA base module: {type(base)}")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.enabled = True
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.reset_parameters()
        self._move_to_base_device()
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def _move_to_base_device(self):
        device = self.base.weight.device
        dtype = self.base.weight.dtype
        self.lora_A.data = self.lora_A.data.to(device=device, dtype=dtype)
        self.lora_B.data = self.lora_B.data.to(device=device, dtype=dtype)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}"

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = F.linear(x, self.lora_A)
        x = F.linear(x, self.lora_B)
        return x * self.scaling

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        if isinstance(x, SparseTensor):
            base_out = self.base(x)
            delta = self._delta(x.feats)
            return base_out.replace(base_out.feats + delta)
        base_out = self.base(x)
        return base_out + self._delta(x)


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def freeze_module(module: nn.Module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def apply_lora(model: nn.Module, spec: LoraSpec) -> List[str]:
    replaced = []
    module_names = [name for name, module in model.named_modules() if isinstance(module, (nn.Linear, SparseLinear))]
    for name in module_names:
        if not name.endswith(spec.target_suffixes):
            continue
        parent, child_name = _get_parent_module(model, name)
        base = getattr(parent, child_name)
        setattr(parent, child_name, LoraLinear(base, rank=spec.rank, alpha=spec.alpha, dropout=spec.dropout))
        replaced.append(name)
    return replaced


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    missing = []
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        key_a = f"{name}.lora_A"
        key_b = f"{name}.lora_B"
        if key_a not in state_dict or key_b not in state_dict:
            missing.append(name)
            continue
        module.lora_A.data.copy_(state_dict[key_a].to(module.lora_A.device, dtype=module.lora_A.dtype))
        module.lora_B.data.copy_(state_dict[key_b].to(module.lora_B.device, dtype=module.lora_B.dtype))
    return missing


def get_lora_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for module in model.modules():
        if isinstance(module, LoraLinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@contextmanager
def disable_lora(model: nn.Module):
    modules = [module for module in model.modules() if isinstance(module, LoraLinear)]
    prev_states = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, prev_states):
            module.enabled = enabled
