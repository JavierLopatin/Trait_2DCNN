"""Pre-compute 2D transformations and save to disk.

This avoids recomputing transforms every epoch, which is the bottleneck on CPU.

Usage:
    python -m training.precompute --transform cwt --output-size 64
"""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .config import TrainConfig, TRAIT_NAMES
from .data_loader import load_fold_data
from transforms import TRANSFORMS


def precompute_fold(transform, spectra: np.ndarray, output_path: Path):
    """Transform all spectra and save as memory-mapped array."""
    # Get output shape from first sample
    sample = transform.transform(spectra[0])
    shape = (len(spectra),) + sample.shape
    dtype = np.float32

    # Create memory-mapped file
    mmap = np.memmap(output_path, dtype=dtype, mode='w+', shape=shape)

    for i in tqdm(range(len(spectra)), desc=f"Transforming"):
        mmap[i] = transform.transform(spectra[i])
        if (i + 1) % 1000 == 0:
            mmap.flush()

    mmap.flush()

    # Save shape metadata
    np.save(str(output_path) + '.shape.npy', np.array(shape))
    print(f"Saved {output_path}: shape={shape}, size={output_path.stat().st_size / 1e6:.1f} MB")
    return shape


def precompute_all(transform_name: str, output_size: int = 64,
                   data_dir: Path = Path('multi-traitretrieval/dataset'),
                   cache_dir: Path = Path('cache')):
    """Pre-compute transforms for all folds."""
    TransformClass = TRANSFORMS[transform_name]
    transform = TransformClass(output_size=output_size)

    cache_base = cache_dir / f'{transform_name}_{output_size}'
    cache_base.mkdir(parents=True, exist_ok=True)

    for fold in range(1, 6):
        print(f"\n--- Fold {fold} ---")
        train_spectra, train_labels, test_spectra, test_labels, _ = \
            load_fold_data(data_dir, fold)

        train_path = cache_base / f'train_fold{fold}.dat'
        test_path = cache_base / f'test_fold{fold}.dat'

        if not train_path.exists():
            precompute_fold(transform, train_spectra, train_path)
        else:
            print(f"  Train fold {fold} already cached")

        if not test_path.exists():
            precompute_fold(transform, test_spectra, test_path)
        else:
            print(f"  Test fold {fold} already cached")

    print(f"\nDone! Cached at: {cache_base}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', type=str, required=True,
                        choices=list(TRANSFORMS.keys()))
    parser.add_argument('--output-size', type=int, default=64)
    parser.add_argument('--data-dir', type=str, default='multi-traitretrieval/dataset')
    parser.add_argument('--cache-dir', type=str, default='cache')
    args = parser.parse_args()

    precompute_all(args.transform, args.output_size,
                   Path(args.data_dir), Path(args.cache_dir))


if __name__ == '__main__':
    main()
