import torch
import numpy as np
import lightning as L
from .losses import MaskedMSELoss, MaskedHuberLoss


class TraitRegressionModule(L.LightningModule):
    """Lightning module for multi-trait regression with masked loss."""

    def __init__(self, model: torch.nn.Module, lr: float = 1e-3,
                 weight_decay: float = 1e-4, loss_type: str = 'mse'):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['model'])

        if loss_type == 'huber':
            self.loss_fn = MaskedHuberLoss()
        else:
            self.loss_fn = MaskedMSELoss()

        # For epoch-level R² computation
        self._val_preds = []
        self._val_targets = []
        self._val_masks = []
        self._test_preds = []
        self._test_targets = []
        self._test_masks = []

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        imgs, labels, mask = batch
        pred = self(imgs)
        loss = self.loss_fn(pred, labels, mask)
        return pred, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, mask = batch
        pred, loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        self._val_preds.append(pred.detach().cpu())
        self._val_targets.append(labels.detach().cpu())
        self._val_masks.append(mask.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds)
        targets = torch.cat(self._val_targets)
        masks = torch.cat(self._val_masks)

        r2_vals = self._compute_r2(preds, targets, masks)
        mean_r2 = np.nanmean(r2_vals)
        self.log('val_r2_mean', mean_r2, prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()
        self._val_masks.clear()

    def test_step(self, batch, batch_idx):
        imgs, labels, mask = batch
        pred, loss = self._step(batch)
        self.log('test_loss', loss, sync_dist=True)

        self._test_preds.append(pred.detach().cpu())
        self._test_targets.append(labels.detach().cpu())
        self._test_masks.append(mask.detach().cpu())

    def on_test_epoch_end(self):
        if not self._test_preds:
            return
        preds = torch.cat(self._test_preds)
        targets = torch.cat(self._test_targets)
        masks = torch.cat(self._test_masks)

        r2_vals = self._compute_r2(preds, targets, masks)
        mean_r2 = np.nanmean(r2_vals)
        self.log('test_r2_mean', mean_r2)

        # Store for external access
        self.test_results = {
            'predictions': preds.numpy(),
            'targets': targets.numpy(),
            'masks': masks.numpy(),
            'r2_per_trait': r2_vals,
        }

        self._test_preds.clear()
        self._test_targets.clear()
        self._test_masks.clear()

    @staticmethod
    def _compute_r2(preds, targets, masks):
        """Compute R² per trait."""
        r2 = np.full(preds.shape[1], np.nan)
        for j in range(preds.shape[1]):
            m = masks[:, j].bool()
            if m.sum() < 2:
                continue
            y = targets[m, j].numpy()
            yhat = preds[m, j].numpy()
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot > 0:
                r2[j] = 1 - ss_res / ss_tot
        return r2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
