"""A linear probe on one layer's residual stream: logit = w . x + b.

Feature standardization from training is folded into (w, b), so serving is
one dot product per token. Thresholds live on the logit, not the probability:
a confident probe pushes fp32 sigmoid to exactly 1.0, which would tie the
very scores a low false-positive threshold has to separate. A probe file is a safetensors file with the
weight tensor and its scalars in the metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass
class LinearProbe:
    layer: int  # reads the residual stream after this decoder layer
    weight: torch.Tensor  # [hidden], fp32
    bias: float
    threshold: float  # logit >= threshold counts as a positive

    def to(self, device: torch.device | str) -> LinearProbe:
        return LinearProbe(self.layer, self.weight.to(device), self.bias, self.threshold)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        """[..., hidden] activations (any dtype) -> [...] fp32 logits."""
        return x.float() @ self.weight + self.bias

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(x))

    @classmethod
    def from_standardized(
        cls, layer: int, coef: torch.Tensor, intercept: float, mean: torch.Tensor, std: torch.Tensor, threshold: float
    ) -> LinearProbe:
        """Fold a model trained on (x - mean) / std into raw-activation weights."""
        weight = (coef / std).float()
        return cls(layer, weight, float(intercept - (weight * mean).sum()), threshold)

    def save(self, path: str | Path) -> None:
        meta = {"layer": str(self.layer), "bias": repr(self.bias), "threshold": repr(self.threshold)}
        save_file({"weight": self.weight.contiguous().cpu()}, str(path), metadata=meta)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> LinearProbe:
        with safe_open(str(path), framework="pt", device=str(device)) as f:
            meta = f.metadata()
            weight = f.get_tensor("weight")
        return cls(int(meta["layer"]), weight, float(meta["bias"]), float(meta["threshold"]))
