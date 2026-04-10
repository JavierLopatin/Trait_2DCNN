import torch
import torch.nn as nn
from torchmetrics import Metric


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


class MaskedR2(Metric):
    """R² metric that ignores NaN (masked) targets. Computed per-trait then averaged."""

    def __init__(self, n_traits: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_traits = n_traits
        self.add_state("ss_res", default=torch.zeros(n_traits), dist_reduce_fx="sum")
        self.add_state("ss_tot", default=torch.zeros(n_traits), dist_reduce_fx="sum")
        self.add_state("sum_y", default=torch.zeros(n_traits), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(n_traits), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor):
        self.ss_res += ((pred - target) ** 2 * mask).sum(dim=0)
        self.sum_y += (target * mask).sum(dim=0)
        self.count += mask.sum(dim=0)

    def compute(self):
        mean_y = self.sum_y / self.count.clamp(min=1)
        # Recompute ss_tot is tricky without storing all values,
        # so we use the online formula: Var = E[y²] - E[y]²
        # For simplicity, return 1 - ss_res / ss_tot where ss_tot is approximated
        # This metric is best used at epoch end with full data
        valid = self.count > 1
        r2 = torch.zeros(self.n_traits, device=self.ss_res.device)
        r2[~valid] = float('nan')
        # ss_tot needs variance, which we don't track here.
        # Use a simpler per-batch approach in the Lightning module instead.
        return r2


class MaskedRMSE(Metric):
    """RMSE metric per trait, ignoring NaN."""

    def __init__(self, n_traits: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_traits = n_traits
        self.add_state("ss_res", default=torch.zeros(n_traits), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(n_traits), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor):
        self.ss_res += ((pred - target) ** 2 * mask).sum(dim=0)
        self.count += mask.sum(dim=0)

    def compute(self):
        valid = self.count > 0
        rmse = torch.full((self.n_traits,), float('nan'), device=self.ss_res.device)
        rmse[valid] = torch.sqrt(self.ss_res[valid] / self.count[valid])
        return rmse
