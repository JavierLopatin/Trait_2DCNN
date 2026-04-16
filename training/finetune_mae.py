"""Fine-tune pretrained MAE-2D encoder for trait regression.

Usage:
    # Fine-tune on real labeled data (Exp 3)
    python -m training.finetune_mae --checkpoint results/mae_pretrained/best.pt --dataset greenhs

    # Fine-tune on PROSAIL, test on real (Exp 5)
    python -m training.finetune_mae --checkpoint results/mae_pretrained/best.pt --dataset prosail --test-dataset greenhs
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from .config import (
    GHS_DATA_DIR, GHS_N_TRAITS, GHS_TRAIT_SHORT,
    DATASET_CHERIF, DATASET_GHS, DATASET_PROSAIL,
)
from .data_loader import (load_ghs_data, fit_scaler, apply_scaler,
                          CachedDataset, load_cached_images)
from .lightning_module import TraitRegressionModule
from .train_2d import evaluate_and_save
from models.mae_2d import MAE2D, MAERegressionModel


def load_mae_encoder(checkpoint_path: str, device='cpu'):
    """Load pretrained MAE and extract encoder."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    mae = MAE2D(
        img_size=224,
        patch_size=16,
        in_chans=1,
        encoder_embed_dim=192,
        encoder_depth=config.get('encoder_depth', 6),
        encoder_num_heads=3,
        decoder_embed_dim=96,
        decoder_depth=config.get('decoder_depth', 4),
        decoder_num_heads=3,
    )
    mae.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded MAE from {checkpoint_path} (epoch {ckpt['epoch']}, "
          f"val_loss {ckpt['val_loss']:.4f})")

    return mae.get_encoder()


def finetune(args):
    L.seed_everything(args.seed)

    # Load pretrained encoder
    encoder = load_mae_encoder(args.checkpoint)
    model = MAERegressionModel(
        encoder, n_traits=GHS_N_TRAITS,
        freeze_encoder=(args.mode == 'linear_probe'))

    mode_desc = 'linear probe' if args.mode == 'linear_probe' else 'fine-tune'
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Mode: {mode_desc}, trainable params: {n_trainable/1e6:.2f}M")

    # Load labels and cached images (single load each)
    data_dir = Path(args.data_dir) if args.data_dir else GHS_DATA_DIR
    cache_suffix = f'_{args.dataset}' if args.dataset != DATASET_CHERIF else ''
    cache_dir = Path('cache') / f'reshape_224{cache_suffix}'

    if not cache_dir.exists() or not (cache_dir / 'train_fold1.dat').exists():
        print(f"ERROR: Cache not found at {cache_dir}")
        print("Run first: python -m training.precompute --transform reshape "
              "--output-size 224 --dataset greenhs")
        return

    if args.test_dataset and args.test_dataset != args.dataset:
        # Exp 5: train on PROSAIL, test on real GHS
        from .data_loader import load_prosail_data, _load_single_cache
        _, train_labels = load_prosail_data(data_dir)
        train_shape = tuple(np.load(str(cache_dir / 'train_fold1.dat') + '.shape.npy'))
        train_images = np.memmap(cache_dir / 'train_fold1.dat',
                                 dtype=np.float32, mode='r', shape=train_shape)
        test_cache_dir = Path('cache') / 'reshape_224_greenhs'
        _, _, _, test_labels, _ = load_ghs_data(GHS_DATA_DIR)
        _, test_imgs = load_cached_images(test_cache_dir, 1)
    else:
        # Exp 3: train and test on real GHS
        _, train_labels, _, test_labels, _ = load_ghs_data(data_dir)
        train_images, test_imgs = load_cached_images(cache_dir, 1)

    # Scaler
    scaler = fit_scaler(train_labels)
    train_labels_scaled = apply_scaler(train_labels, scaler)
    test_labels_scaled = apply_scaler(test_labels, scaler)

    # Train/val split
    idx = np.arange(len(train_labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.15, random_state=args.seed)

    train_ds = CachedDataset(
        train_images[train_idx], train_labels_scaled[train_idx], augment=True)
    val_ds = CachedDataset(
        train_images[val_idx], train_labels_scaled[val_idx], augment=False)
    test_ds = CachedDataset(test_imgs, test_labels_scaled, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Lightning module
    lr = args.lr_head if args.mode == 'linear_probe' else args.lr
    lit_model = TraitRegressionModule(model, lr=lr, weight_decay=args.weight_decay)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, mode='min'),
        ModelCheckpoint(
            dirpath=output_dir, filename='best',
            monitor='val_loss', mode='min', save_top_k=1),
    ]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        default_root_dir=output_dir,
    )

    trainer.fit(lit_model, train_loader, val_loader)
    trainer.test(lit_model, test_loader, ckpt_path='best')

    # Save scaler
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Compute real-scale metrics (reuse shared evaluation logic)
    results = lit_model.test_results
    metrics = evaluate_and_save(results, scaler, GHS_TRAIT_SHORT, output_dir)

    metrics_df = pd.read_csv(output_dir / 'metrics.csv', index_col=0)
    print(f"\nResults ({mode_desc}):")
    print(metrics_df[['R2', 'RMSE']].to_string())
    print(f"\nMean R²: {np.nanmean(metrics['R2']):.4f}")

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune MAE-2D for traits')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained MAE checkpoint')
    parser.add_argument('--dataset', type=str, default=DATASET_GHS,
                        choices=[DATASET_GHS, DATASET_PROSAIL])
    parser.add_argument('--test-dataset', type=str, default=None,
                        help='Test on different dataset (e.g., greenhs when training on prosail)')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='finetune',
                        choices=['finetune', 'linear_probe'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--lr-head', type=float, default=1e-3,
                        help='Learning rate for linear probe')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=155)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'results/mae_{args.mode}_{args.dataset}'

    finetune(args)


if __name__ == '__main__':
    main()
