import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()
