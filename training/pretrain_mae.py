"""MAE-2D pretraining on unlabeled spectral images.

Usage:
    python -m training.pretrain_mae --epochs 300 --batch-size 64
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.mae_2d import MAE2D
from .data_loader import UnlabeledCachedDataset


def pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load precomputed unlabeled images
    cache_dir = Path(args.cache_dir)
    data_path = cache_dir / 'unlabeled.dat'
    shape_path = str(data_path) + '.shape.npy'

    if not data_path.exists():
        print(f"ERROR: Cache not found at {data_path}")
        print("Run first: python -m training.precompute --transform reshape "
              "--output-size 224 --dataset greenhs_unlabeled")
        return

    shape = tuple(np.load(shape_path))
    images = np.memmap(data_path, dtype=np.float32, mode='r', shape=shape)
    print(f"Loaded {len(images)} unlabeled images, shape: {shape}")

    # Filter out all-NaN images (some unlabeled spectra have missing data)
    print("Filtering NaN images...")
    valid_mask = np.array([
        not np.isnan(images[i].flat[0]) for i in range(len(images))
    ])
    valid_indices = np.where(valid_mask)[0]
    n_bad = len(images) - len(valid_indices)
    if n_bad > 0:
        print(f"  Removed {n_bad} all-NaN images, {len(valid_indices)} remain")

    # Split into train/val (95/5)
    n_val = max(int(len(valid_indices) * 0.05), 1000)
    shuffled = np.random.RandomState(42).permutation(valid_indices)
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]

    # No augmentation for MAE pretraining — random masking is sufficient
    # (consistent with He et al. 2022 and Cherif et al. 2025)
    train_ds = UnlabeledCachedDataset.from_memmap(images, indices=train_idx, augment=False)
    val_ds = UnlabeledCachedDataset.from_memmap(images, indices=val_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # Create MAE model
    model = MAE2D(
        img_size=224,
        patch_size=16,
        in_chans=1,
        encoder_embed_dim=192,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=3,
        decoder_embed_dim=96,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=3,
        mask_ratio=args.mask_ratio,
        cosine_weight=args.cosine_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MAE-2D: {n_params:.1f}M params")
    print(f"  Encoder: depth={args.encoder_depth}, heads=3, dim=192")
    print(f"  Decoder: depth={args.decoder_depth}, heads=3, dim=96")
    print(f"  Mask ratio: {args.mask_ratio}")
    print(f"  Cosine weight: {args.cosine_weight}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            imgs = batch.to(device)
            loss, _, _ = model(imgs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch.to(device)
                loss, _, _ = model(imgs)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} — "
                  f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                  f"lr: {lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': vars(args),
            }, output_dir / 'best.pt')
        else:
            patience_counter += 1

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(no improvement for {args.patience} epochs)")
            break

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch{epoch+1}.pt')

    print(f"\nPretraining complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(description='MAE-2D pretraining')
    parser.add_argument('--cache-dir', type=str,
                        default='cache/reshape_224_greenhs_unlabeled')
    parser.add_argument('--output-dir', type=str,
                        default='results/mae_pretrained')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--mask-ratio', type=float, default=0.75)
    parser.add_argument('--cosine-weight', type=float, default=1.0)
    parser.add_argument('--encoder-depth', type=int, default=6)
    parser.add_argument('--decoder-depth', type=int, default=4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pretrain(args)


if __name__ == '__main__':
    main()
