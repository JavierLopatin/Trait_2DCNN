"""Main training script for 2D spectral trait regression.

Usage:
    python -m training.train_2d --transform cwt --model efficientnet_b0 --epochs 100
    python -m training.train_2d --transform cos2d --model vit_small --epochs 100
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .config import (
    TrainConfig, DATASET_CHERIF, DATASET_GHS, default_data_dir,
)
from .data_loader import create_dataloaders
from .lightning_module import TraitRegressionModule
from transforms import TRANSFORMS, CompositeTransform
from models.trait_model import get_model, MODEL_REGISTRY


def make_transform(name: str, output_size: int):
    """Create a transform from a name, supporting '+' for composites.

    Examples:
        'cwt' → CWTTransform(output_size)
        'reshape+cwt' → CompositeTransform([ReshapeTransform, CWTTransform])
        'reshape+cwt+spectrogram' → CompositeTransform([...])
    """
    parts = name.split('+')
    if len(parts) == 1:
        return TRANSFORMS[name](output_size=output_size)
    sub_transforms = [TRANSFORMS[p](output_size=output_size) for p in parts]
    return CompositeTransform(sub_transforms)


def train_fold(config: TrainConfig, fold: int, transform, model_fn,
               cache_dir: Path = None):
    """Train and evaluate a single fold (or single run for GHS)."""
    L.seed_everything(config.seed + fold)

    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = \
        create_dataloaders(config, fold, transform, cache_dir=cache_dir)

    # Create model
    model = model_fn()
    lit_model = TraitRegressionModule(
        model, lr=config.lr, weight_decay=config.weight_decay)

    # Output directory
    fold_dir = config.output_dir / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.patience, mode='min'),
        ModelCheckpoint(
            dirpath=fold_dir, filename='best',
            monitor='val_loss', mode='min', save_top_k=1),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        default_root_dir=fold_dir,
    )

    # Train
    trainer.fit(lit_model, train_loader, val_loader)

    # Test with best checkpoint
    trainer.test(lit_model, test_loader, ckpt_path='best')

    # Save scaler and results
    with open(fold_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    results = lit_model.test_results
    metrics = evaluate_and_save(results, scaler, config.trait_short, fold_dir)

    print(f"\nFold {fold} — Mean R²: {np.nanmean(metrics['R2']):.4f}, "
          f"Mean RMSE: {np.nanmean(metrics['RMSE']):.4f}")

    return metrics


def evaluate_and_save(results: dict, scaler, trait_short: list,
                      output_dir: Path) -> dict:
    """Inverse-transform predictions, compute metrics, and save to disk.

    Args:
        results: dict from TraitRegressionModule.test_results
                 (keys: predictions, targets, masks)
        scaler: fitted PowerTransformer for inverse scaling
        trait_short: list of short trait names for column labels
        output_dir: directory to save metrics.csv and predictions.csv

    Returns:
        metrics: dict with R2, RMSE, nRMSE, MAE, Bias arrays
    """
    preds_real = scaler.inverse_transform(results['predictions'])
    targets_real = scaler.inverse_transform(
        np.nan_to_num(results['targets'], nan=0.0))
    targets_real[~results['masks'].astype(bool)] = np.nan

    metrics = compute_metrics(preds_real, targets_real, results['masks'])
    trait_labels = trait_short[:len(metrics['R2'])]
    metrics_df = pd.DataFrame(metrics, index=trait_labels)
    metrics_df.to_csv(output_dir / 'metrics.csv')

    pd.DataFrame(
        preds_real, columns=trait_short[:preds_real.shape[1]]
    ).to_csv(output_dir / 'predictions.csv', index=False)

    return metrics


def compute_metrics(preds, targets, masks):
    """Compute R², RMSE, nRMSE, MAE, Bias per trait."""
    n_traits = preds.shape[1]
    r2 = np.full(n_traits, np.nan)
    rmse = np.full(n_traits, np.nan)
    nrmse = np.full(n_traits, np.nan)
    mae = np.full(n_traits, np.nan)
    bias = np.full(n_traits, np.nan)

    for j in range(n_traits):
        m = masks[:, j].astype(bool)
        if m.sum() < 2:
            continue
        y = targets[m, j]
        yhat = preds[m, j]

        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot > 0:
            r2[j] = 1 - ss_res / ss_tot

        rmse[j] = np.sqrt(np.mean((y - yhat) ** 2))
        nrmse[j] = rmse[j] / (y.max() - y.min()) if (y.max() - y.min()) > 0 else np.nan
        mae[j] = np.mean(np.abs(y - yhat))
        bias[j] = np.mean(yhat - y)

    return {'R2': r2, 'RMSE': rmse, 'nRMSE': nrmse, 'MAE': mae, 'Bias': bias}


def run_experiment(config: TrainConfig, start_fold: int = 1):
    """Run full 5-fold CV experiment."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / 'config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(config).items()}, f, indent=2)

    # Create transform (supports composite via '+' syntax)
    transform = make_transform(config.transform_name, config.output_size)

    # Model factory
    in_channels = transform.n_channels
    n_traits = config.n_traits
    def model_fn():
        return get_model(config.model_name, in_channels=in_channels,
                         n_traits=n_traits, pretrained=config.pretrained)

    # Check for cached transforms
    cache_suffix = f'_{config.dataset}' if config.dataset != DATASET_CHERIF else ''
    parts = config.transform_name.split('+')
    is_composite = len(parts) > 1
    if is_composite:
        cache_dirs = [Path('cache') / f'{p}_{config.output_size}{cache_suffix}'
                      for p in parts]
        use_cache = all(d.exists() and (d / 'train_fold1.dat').exists()
                        for d in cache_dirs)
        cache_dir = cache_dirs if use_cache else None
    else:
        cache_dir = Path('cache') / f'{config.transform_name}_{config.output_size}{cache_suffix}'
        use_cache = cache_dir.exists() and (cache_dir / 'train_fold1.dat').exists()
        cache_dir = cache_dir if use_cache else None

    print(f"Experiment: transform={config.transform_name}, model={config.model_name}")
    print(f"  Input channels: {in_channels}, Output size: {config.output_size}")
    print(f"  Cache: {'YES' if use_cache else 'NO (on-the-fly)'}")
    print(f"  Output dir: {config.output_dir}")
    print()

    all_metrics = []
    for fold in range(1, config.n_folds + 1):
        fold_dir = config.output_dir / f'fold_{fold}'
        # Skip completed folds (those with metrics.csv)
        if fold < start_fold or (fold_dir / 'metrics.csv').exists():
            print(f"Skipping fold {fold} (already completed)")
            # Load existing metrics for aggregation
            metrics_df = pd.read_csv(fold_dir / 'metrics.csv', index_col=0)
            all_metrics.append({col: metrics_df[col].values for col in metrics_df.columns})
            continue
        print(f"{'='*60}")
        print(f"FOLD {fold}/{config.n_folds}")
        print(f"{'='*60}")
        metrics = train_fold(config, fold, transform, model_fn,
                             cache_dir=cache_dir)
        all_metrics.append(metrics)

    # Aggregate metrics across folds
    agg = {}
    for key in all_metrics[0]:
        vals = np.stack([m[key] for m in all_metrics])
        agg[f'{key}_mean'] = np.nanmean(vals, axis=0)
        agg[f'{key}_std'] = np.nanstd(vals, axis=0)

    summary = pd.DataFrame(agg, index=config.trait_short[:len(agg['R2_mean'])])
    summary.to_csv(config.output_dir / 'summary.csv')

    print(f"\n{'='*60}")
    print("SUMMARY (mean ± std across folds)")
    print(f"{'='*60}")
    print(summary[['R2_mean', 'R2_std', 'RMSE_mean']].to_string())
    print(f"\nOverall Mean R²: {np.nanmean(agg['R2_mean']):.4f} "
          f"± {np.nanmean(agg['R2_std']):.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Train 2D spectral trait regression')
    parser.add_argument('--transform', type=str, default='cwt',
                        help='Transform name or composite (e.g. reshape+cwt)')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=DATASET_CHERIF,
                        choices=[DATASET_CHERIF, DATASET_GHS],
                        help='Dataset: cherif2023 (20 traits, 5-fold CV) or greenhs (8 traits, fixed split)')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--precompute', action='store_true',
                        help='Pre-compute transforms before training')
    parser.add_argument('--start-fold', type=int, default=1,
                        help='Start from this fold (skip earlier folds)')
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = str(default_data_dir(args.dataset))

    config = TrainConfig(
        data_dir=Path(data_dir),
        dataset=args.dataset,
        transform_name=args.transform,
        model_name=args.model,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_size=args.output_size,
        seed=args.seed,
        pretrained=args.pretrained,
        num_workers=args.num_workers,
    )

    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    else:
        ds_prefix = f'{args.dataset}_' if args.dataset != DATASET_CHERIF else ''
        config.output_dir = Path('results') / f'{ds_prefix}{args.transform}_{args.model}'

    # Pre-compute transforms if requested
    if args.precompute:
        from .precompute import precompute_all
        for part in args.transform.split('+'):
            precompute_all(part, args.output_size,
                           config.data_dir, Path('cache'))

    run_experiment(config, start_fold=args.start_fold)


if __name__ == '__main__':
    main()
