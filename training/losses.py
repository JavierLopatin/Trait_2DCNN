import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """MSE loss that ignores NaN (masked) targets."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        diff = (pred - target) ** 2
        diff = diff * mask
        count = mask.sum()
        if count == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return diff.sum() / count


class MaskedHuberLoss(nn.Module):
    """Huber loss that ignores NaN (masked) targets."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.delta,
                           0.5 * diff ** 2,
                           self.delta * (diff - 0.5 * self.delta))
        loss = loss * mask
        count = mask.sum()
        if count == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return loss.sum() / count
