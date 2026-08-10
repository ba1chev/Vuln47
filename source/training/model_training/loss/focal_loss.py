import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Class-weighted focal loss for the imbalanced positive class.

    On top of the per-class 'alpha' weight, the '(1 - p_t) ** gamma' factor
    down-weights easy, already-confident examples so training keeps focusing on
    the hard, still-misclassified functions. 'gamma=0' reduces to plain
    weighted cross-entropy.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        # register as a buffer so alpha moves with .to(device) but isn't a learned param
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)                       # pt = predicted prob of the true class
        focal = (1.0 - pt) ** self.gamma * ce     # shrink loss on confident/easy samples
        return focal.mean()
