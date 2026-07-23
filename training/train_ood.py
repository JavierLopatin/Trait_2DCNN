"""Out-of-distribution (cross-dataset) evaluation of the Reshape 2D-CNN.

Replicates the OOD data protocol of Cherif et al. 2025 (GreenHyperSpectra, Table 4)
exactly -- folds, spectral preprocessing, 80/20 split, Box-Cox scaling, masked-Huber
NaN handling, and the sub-sampled macro metric -- and swaps only the model to our
Reshape -> EfficientNet-B0. This isolates the 1D-vs-2D input representation for a 1:1
comparison against their supervised 1D baseline (avg R^2 = 0.243).

Protocol code is ported in training/ood_cherif2025.py (attributed to
https://github.com/echerif18/HyspectraSSL).

Usage (pin GPU 1):
    CUDA_VISIBLE_DEVICES=1 python -m training.train_ood --seed 42
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transforms import ReshapeTransform
from models.trait_model import get_model
from models.mae_2d import MAERegressionModel
from training.finetune_mae import load_mae_encoder
from training.ood_cherif2025 import (
    GHS_OOD_TRAITS, sliding_custom_cv, data_prep_db, drop_spectral_artifacts,
    fit_boxcox_scaler, SpectraAugmenter, HuberCustomLoss, eval_metrics,
    subsample_macro_metrics,
)


class OODReshapeDataset(Dataset):
    """1D spectrum -> (optional Cherif 1D augmentation) -> Reshape -> 2D image.

    Augmentation is random per epoch, so images are produced on the fly (no cache).
    Labels are already Box-Cox transformed (NaN preserved for the masked loss).
    """

    def __init__(self, X, y, augmenter=None, output_size=224):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.aug = augmenter
        self.reshape = ReshapeTransform(output_size=output_size)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.aug is not None:
            x = self.aug(x)
        img = self.reshape.transform(x)                       # (H, W) float32 in [0,1]
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(self.y[i])


def build_model(args, device):
    """Reshape 2D-CNN (efficientnet_b0) or MAE-2D fine-tuning (pretrained encoder +
    fresh regression head, fully trainable). Same 2D reshape input for both."""
    if args.model == 'mae':
        encoder = load_mae_encoder(args.mae_checkpoint, device=device)
        model = MAERegressionModel(encoder, n_traits=len(GHS_OOD_TRAITS),
                                   freeze_encoder=False)
        return model.to(device)
    return get_model(args.model, in_channels=1, n_traits=len(GHS_OOD_TRAITS)).to(device)


def _predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def _val_loss(model, loader, criterion, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            loss = criterion(yb.to(device), model(xb.to(device)))
            tot += float(loss); n += 1
    return tot / max(n, 1)


def train_one_fold(df_train, df_test, args, device):
    """Train on df_train (internal 80/20), return (obs_df, pred_df) on the held-out
    df_test in original trait units."""
    # Preprocess + drop the red-edge artifact samples (Cherif 2025).
    X, y = data_prep_db(df_train, GHS_OOD_TRAITS)
    meta = df_train[['dataset']].loc[X.index]
    X, y, meta = drop_spectral_artifacts(X, y, meta)

    # Internal 80/20 split, stratified by source dataset (Cherif 2025 random_state=300).
    Xtr, Xval = train_test_split(X, test_size=0.2, stratify=meta.dataset, random_state=300)
    ytr, yval = y.loc[Xtr.index], y.loc[Xval.index]

    # Box-Cox scaler fit on TRAIN traits only (NaN-native).
    scaler = fit_boxcox_scaler(ytr)
    ytr_t = scaler.transform(np.asarray(ytr))
    yval_t = scaler.transform(np.asarray(yval))

    train_ds = OODReshapeDataset(Xtr.values, ytr_t, augmenter=SpectraAugmenter(aug_prob=0.7))
    val_ds = OODReshapeDataset(Xval.values, yval_t, augmenter=None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = build_model(args, device)
    criterion = HuberCustomLoss(threshold=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val, best_state, patience = float('inf'), None, 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(yb.to(device), model(xb.to(device)))
            loss.backward()
            optimizer.step()
        scheduler.step()
        vl = _val_loss(model, val_loader, criterion, device)
        if vl < best_val - 1e-5:
            best_val, best_state, patience = vl, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict the held-out datasets (no augmentation) and inverse Box-Cox.
    Xte, yte = data_prep_db(df_test, GHS_OOD_TRAITS)
    test_ds = OODReshapeDataset(Xte.values, np.zeros((len(Xte), len(GHS_OOD_TRAITS)), np.float32),
                                augmenter=None)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    pred_t = _predict(model, test_loader, device)
    pred = scaler.inverse_transform(pred_t)

    obs_df = pd.DataFrame(np.asarray(yte), columns=GHS_OOD_TRAITS)
    pred_df = pd.DataFrame(pred, columns=GHS_OOD_TRAITS)
    ds_ids = df_test['dataset'].values
    return obs_df, pred_df, ds_ids, best_val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='data/GreenHyperSpectra/labeled_all.csv')
    ap.add_argument('--model', default='efficientnet_b0',
                    help="'efficientnet_b0' (supervised) or 'mae' (MAE-2D fine-tuning)")
    ap.add_argument('--mae-checkpoint', default='results/mae_pretrained/best.pt',
                    help='Pretrained MAE-2D checkpoint (used when --model mae)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--out', default='results/ood_reshape')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  visible GPUs={torch.cuda.device_count()}")

    db = pd.read_csv(args.data, low_memory=False)
    if 'Unnamed: 0' in db.columns:
        db = db.drop(['Unnamed: 0'], axis=1)

    obs_all, pred_all, ds_all = [], [], []
    per_fold = []
    for gp, (df_train, df_test, test_ids) in enumerate(sliding_custom_cv(db, seed=args.seed)):
        t0 = time.time()
        obs_df, pred_df, ds_ids, best_val = train_one_fold(df_train, df_test, args, device)
        obs_all.append(obs_df); pred_all.append(pred_df); ds_all.append(ds_ids)
        fold_metrics = eval_metrics(obs_df, pred_df)
        r2 = float(fold_metrics['r2_score'].mean())
        per_fold.append({'fold': gp, 'held_out': str(test_ids), 'n_test': len(df_test),
                         'val_loss': best_val, 'r2_mean_pooled': r2})
        print(f"[fold {gp:2d}] held-out={test_ids} n={len(df_test)} "
              f"val_loss={best_val:.4f} pooled R2={r2:.3f} ({time.time()-t0:.0f}s)")

    obs_cat = pd.concat(obs_all, ignore_index=True)
    pred_cat = pd.concat(pred_all, ignore_index=True)
    ds_cat = np.concatenate(ds_all)
    obs_cat.to_csv(out / 'ood_obs.csv', index=False)
    pred_cat.to_csv(out / 'ood_pred.csv', index=False)
    pd.DataFrame({'dataset': ds_cat}).to_csv(out / 'ood_dataset_ids.csv', index=False)
    pd.DataFrame(per_fold).to_csv(out / 'ood_per_fold.csv', index=False)

    # Cherif 2025 OOD macro metric: 5 subsamples, <=30 samples/dataset.
    mean_df, std_df = subsample_macro_metrics(obs_cat, pred_cat, ds_cat, n_sub=5, max_n=30,
                                              seed=args.seed)
    mean_df.to_csv(out / 'ood_metrics_mean.csv')
    std_df.to_csv(out / 'ood_metrics_std.csv')

    print("\n=== OOD cross-dataset (macro, <=30/dataset x5) — R2 mean ± std ===")
    for tr in GHS_OOD_TRAITS:
        print(f"  {tr:5s}  {mean_df.loc[tr,'r2_score']:.3f} ± {std_df.loc[tr,'r2_score']:.3f}"
              f"   nRMSE {mean_df.loc[tr,'nRMSE (%)']:.2f}")
    print(f"  AVG    {mean_df['r2_score'].mean():.3f}   "
          f"(Cherif 2025 supervised 1D = 0.243)")
    print(f"\nSaved to {out}/")


if __name__ == '__main__':
    main()
