"""Pre-compute 2D transformations and save to disk.

This avoids recomputing transforms every epoch, which is the bottleneck on CPU.

Usage:
    python -m training.precompute --transform cwt --output-size 64
"""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .config import (
    DATASET_CHERIF, DATASET_GHS, DATASET_GHS_UNLABELED, DATASET_PROSAIL,
    default_data_dir,
)
from .data_loader import load_fold_data, load_ghs_data, load_ghs_unlabeled, load_prosail_data
from transforms import TRANSFORMS


def precompute_fold(transform, spectra: np.ndarray, output_path: Path):
    """Transform all spectra and save as memory-mapped array."""
    sample = transform.transform(spectra[0])
    shape = (len(spectra),) + sample.shape
    dtype = np.float32

    mmap = np.memmap(output_path, dtype=dtype, mode='w+', shape=shape)

    for i in tqdm(range(len(spectra)), desc="Transforming"):
        mmap[i] = transform.transform(spectra[i])
        if (i + 1) % 1000 == 0:
            mmap.flush()

    mmap.flush()
    np.save(str(output_path) + '.shape.npy', np.array(shape))
    print(f"Saved {output_path}: shape={shape}, size={output_path.stat().st_size / 1e6:.1f} MB")
    return shape


def _cache_if_missing(transform, spectra: np.ndarray, path: Path, label: str):
    """Precompute and cache spectra if the cache file does not exist."""
    if not path.exists():
        precompute_fold(transform, spectra, path)
    else:
        print(f"  {label} already cached")


def _precompute_ghs(transform, data_dir: Path, cache_base: Path):
    """Precompute GHS labeled train/test data."""
    print("\n--- GreenHyperSpectra (labeled) ---")
    train_spectra, _, test_spectra, _, _ = load_ghs_data(data_dir)
    _cache_if_missing(transform, train_spectra, cache_base / 'train_fold1.dat', 'Train')
    _cache_if_missing(transform, test_spectra, cache_base / 'test_fold1.dat', 'Test')


def _precompute_ghs_unlabeled(transform, data_dir: Path, cache_base: Path):
    """Precompute GHS unlabeled spectra for MAE pretraining."""
    print("\n--- GreenHyperSpectra (unlabeled, 139K) ---")
    spectra = load_ghs_unlabeled(data_dir)
    _cache_if_missing(transform, spectra, cache_base / 'unlabeled.dat', 'Unlabeled')


def _precompute_prosail(transform, data_dir: Path, cache_base: Path):
    """Precompute PROSAIL LUT."""
    print("\n--- PROSAIL LUT ---")
    spectra, _ = load_prosail_data(data_dir)
    _cache_if_missing(transform, spectra, cache_base / 'train_fold1.dat', 'PROSAIL')


def _precompute_cherif(transform, data_dir: Path, cache_base: Path):
    """Precompute Cherif 2023 5-fold CV data."""
    for fold in range(1, 6):
        print(f"\n--- Fold {fold} ---")
        train_spectra, _, test_spectra, _, _ = load_fold_data(data_dir, fold)
        _cache_if_missing(
            transform, train_spectra,
            cache_base / f'train_fold{fold}.dat', f'Train fold {fold}')
        _cache_if_missing(
            transform, test_spectra,
            cache_base / f'test_fold{fold}.dat', f'Test fold {fold}')


# Dispatch table: dataset -> precompute function
_PRECOMPUTE_DISPATCH = {
    DATASET_GHS: _precompute_ghs,
    DATASET_GHS_UNLABELED: _precompute_ghs_unlabeled,
    DATASET_PROSAIL: _precompute_prosail,
    DATASET_CHERIF: _precompute_cherif,
}


def precompute_all(transform_name: str, output_size: int = 64,
                   data_dir: Path = Path('multi-traitretrieval/dataset'),
                   cache_dir: Path = Path('cache'),
                   dataset: str = DATASET_CHERIF):
    """Pre-compute transforms for all folds (or single split for GHS)."""
    TransformClass = TRANSFORMS[transform_name]
    transform = TransformClass(output_size=output_size)

    suffix = f'_{dataset}' if dataset != DATASET_CHERIF else ''
    cache_base = cache_dir / f'{transform_name}_{output_size}{suffix}'
    cache_base.mkdir(parents=True, exist_ok=True)

    dispatch_fn = _PRECOMPUTE_DISPATCH.get(dataset, _precompute_cherif)
    dispatch_fn(transform, data_dir, cache_base)

    print(f"\nDone! Cached at: {cache_base}")


def main():
    all_datasets = list(_PRECOMPUTE_DISPATCH.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', type=str, required=True,
                        choices=list(TRANSFORMS.keys()))
    parser.add_argument('--output-size', type=int, default=64)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default='cache')
    parser.add_argument('--dataset', type=str, default=DATASET_CHERIF,
                        choices=all_datasets)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir(args.dataset)

    precompute_all(args.transform, args.output_size,
                   data_dir, Path(args.cache_dir), args.dataset)


if __name__ == '__main__':
    main()
