import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

from .config import TRAIT_NAMES, GHS_TRAIT_NAMES, DATASET_GHS, TrainConfig
from transforms.base import BaseTransform


class CachedDataset(Dataset):
    """Dataset that loads pre-computed 2D images from memory-mapped files."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 augment: bool = False):
        self.images = images
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)

        # Light augmentation on the image: random noise
        if self.augment:
            img = img + np.random.normal(0, 0.01, img.shape).astype(np.float32)

        # Ensure channel dimension: (H, W) → (1, H, W) or (H, W, C) → (C, H, W)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        else:
            img = img.transpose(2, 0, 1)

        labels = self.labels[idx]
        mask = ~np.isnan(labels)
        labels = np.nan_to_num(labels, nan=0.0)

        return (torch.from_numpy(img),
                torch.from_numpy(labels),
                torch.from_numpy(mask.astype(np.float32)))


class UnlabeledCachedDataset(Dataset):
    """Dataset for unlabeled pre-computed 2D images (MAE pretraining).

    Supports index-based access into a memmap to avoid materializing
    large subsets into RAM when constructed with fancy indexing.
    """

    def __init__(self, images: np.ndarray, indices: np.ndarray = None,
                 augment: bool = False):
        self.images = images
        self.indices = indices
        self.augment = augment

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.images)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        img = self.images[real_idx].astype(np.float32)

        if self.augment:
            img = img + np.random.normal(0, 0.01, img.shape).astype(np.float32)

        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        else:
            img = img.transpose(2, 0, 1)

        return torch.from_numpy(img)


class CompositeCachedDataset(Dataset):
    """Dataset that lazily stacks channels from multiple memmap arrays.

    Keeps references to the original memmaps and an index array.
    Each sample's channels are stacked on-the-fly in __getitem__,
    avoiding materializing the full concatenated array in RAM.
    """

    def __init__(self, image_parts: list, indices: np.ndarray,
                 labels: np.ndarray, augment: bool = False):
        self.image_parts = image_parts  # list of memmap arrays
        self.indices = indices  # index into the memmaps
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        channels = []
        for part in self.image_parts:
            img = part[real_idx].astype(np.float32)
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            channels.append(img)
        img = np.concatenate(channels, axis=-1)

        if self.augment:
            img = img + np.random.normal(0, 0.01, img.shape).astype(np.float32)

        # (H, W, C) → (C, H, W)
        img = img.transpose(2, 0, 1)

        labels = self.labels[idx]
        mask = ~np.isnan(labels)
        labels = np.nan_to_num(labels, nan=0.0)

        return (torch.from_numpy(img),
                torch.from_numpy(labels),
                torch.from_numpy(mask.astype(np.float32)))


class OnTheFlyDataset(Dataset):
    """Dataset that applies 1D→2D transform on-the-fly (slower, for testing)."""

    def __init__(self, spectra: np.ndarray, labels: np.ndarray,
                 transform: BaseTransform, augment: bool = False,
                 aug_shift: float = 0.02, aug_mult: float = 0.02):
        self.spectra = spectra.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.transform = transform
        self.augment = augment
        self.aug_shift = aug_shift
        self.aug_mult = aug_mult

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx].copy()

        if self.augment:
            spectrum = spectrum + np.random.uniform(-self.aug_shift, self.aug_shift)
            spectrum = spectrum * np.random.uniform(1 - self.aug_mult, 1 + self.aug_mult)

        img = self.transform.transform(spectrum)

        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        else:
            img = img.transpose(2, 0, 1)

        labels = self.labels[idx]
        mask = ~np.isnan(labels)
        labels = np.nan_to_num(labels, nan=0.0)

        return (torch.from_numpy(img),
                torch.from_numpy(labels),
                torch.from_numpy(mask.astype(np.float32)))


def load_fold_data(data_dir: Path, fold: int):
    """Load train and test data for a given CV fold (Cherif 2023)."""
    train_df = pd.read_csv(data_dir / f'fillCV_{fold}.csv', index_col=0)
    test_df = pd.read_csv(data_dir / f'testCV_{fold}.csv', index_col=0)

    spec_cols = [c for c in train_df.columns if c not in TRAIT_NAMES]
    trait_cols = [c for c in TRAIT_NAMES if c in train_df.columns]

    train_spectra = train_df[spec_cols].values
    train_labels = train_df[trait_cols].values
    test_spectra = test_df[spec_cols].values

    test_trait_cols = [c for c in trait_cols if c in test_df.columns]
    test_labels = test_df[test_trait_cols].values

    sw_path = data_dir / f'samp_w_tr_{fold}.csv'
    sample_weights = None
    if sw_path.exists():
        sample_weights = pd.read_csv(sw_path, index_col=0).values.flatten()

    return train_spectra, train_labels, test_spectra, test_labels, sample_weights


def _split_trait_spectral_columns(df: pd.DataFrame, trait_names: list):
    """Split a DataFrame into trait and spectral columns.

    Returns:
        trait_cols: list of trait column names found in df
        spec_cols: list of spectral column names (everything else, excluding 'dataset')
    """
    trait_cols = [c for c in trait_names if c in df.columns]
    spec_cols = [c for c in df.columns
                 if c not in trait_cols and c != 'dataset']
    return trait_cols, spec_cols


def _extract_spectra_and_labels(df: pd.DataFrame, trait_names: list):
    """Extract spectra and labels arrays from a DataFrame.

    Returns:
        spectra: np.ndarray of shape (n_samples, n_bands), float32
        labels: np.ndarray of shape (n_samples, n_traits), float32
    """
    trait_cols, spec_cols = _split_trait_spectral_columns(df, trait_names)
    spectra = df[spec_cols].values.astype(np.float32)
    labels = df[trait_cols].values.astype(np.float32)
    return spectra, labels


def load_ghs_data(data_dir: Path):
    """Load GreenHyperSpectra train/test data (fixed split, 8 traits)."""
    train_df = pd.read_csv(data_dir / 'labeled_train.csv')
    test_df = pd.read_csv(data_dir / 'labeled_test.csv')

    train_spectra, train_labels = _extract_spectra_and_labels(train_df, GHS_TRAIT_NAMES)
    test_spectra, test_labels = _extract_spectra_and_labels(test_df, GHS_TRAIT_NAMES)

    return train_spectra, train_labels, test_spectra, test_labels, None


def load_ghs_unlabeled(data_dir: Path):
    """Load GreenHyperSpectra unlabeled spectra for pretraining."""
    df = pd.read_parquet(data_dir / 'unlabeled.parquet')
    return df.values.astype(np.float32)


def load_prosail_data(data_dir: Path):
    """Load PROSAIL LUT as training data (test data comes from GHS)."""
    df = pd.read_csv(data_dir / 'prosail_lut.csv')
    spectra, labels = _extract_spectra_and_labels(df, GHS_TRAIT_NAMES)
    return spectra, labels


def load_cached_images(cache_dir, fold: int):
    """Load pre-computed images from memory-mapped files.

    Args:
        cache_dir: Path or list of Paths. If a list, returns separate
                   memmap arrays (for lazy stacking in CompositeCachedDataset).
        fold: CV fold number.

    Returns:
        If single cache_dir: (train_images, test_images) as memmaps.
        If list of cache_dirs: (train_parts, test_parts) as lists of memmaps.
    """
    if isinstance(cache_dir, (list, tuple)):
        train_parts, test_parts = [], []
        for cd in cache_dir:
            t, te = _load_single_cache(cd, fold)
            train_parts.append(t)
            test_parts.append(te)
        return train_parts, test_parts

    return _load_single_cache(cache_dir, fold)


def _load_single_cache(cache_dir: Path, fold: int):
    """Load a single cache directory."""
    train_path = cache_dir / f'train_fold{fold}.dat'
    test_path = cache_dir / f'test_fold{fold}.dat'

    train_shape = tuple(np.load(str(train_path) + '.shape.npy'))
    test_shape = tuple(np.load(str(test_path) + '.shape.npy'))

    train_images = np.memmap(train_path, dtype=np.float32, mode='r', shape=train_shape)
    test_images = np.memmap(test_path, dtype=np.float32, mode='r', shape=test_shape)

    return train_images, test_images


def fit_scaler(labels: np.ndarray):
    """Fit PowerTransformer (Yeo-Johnson) per trait, handling NaN."""
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    labels_filled = labels.copy()
    for j in range(labels.shape[1]):
        col = labels[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            labels_filled[np.isnan(col), j] = np.median(valid)
        else:
            labels_filled[np.isnan(col), j] = 0
    scaler.fit(labels_filled)
    return scaler


def apply_scaler(labels: np.ndarray, scaler: PowerTransformer) -> np.ndarray:
    """Apply scaler while preserving NaN positions."""
    mask = np.isnan(labels)
    labels_filled = labels.copy()
    labels_filled[mask] = 0
    transformed = scaler.transform(labels_filled)
    transformed[mask] = np.nan
    return transformed


def create_dataloaders(config: TrainConfig, fold: int,
                       transform: BaseTransform = None,
                       cache_dir: Path = None):
    """Create train, val, and test DataLoaders for a fold.

    If cache_dir is provided, loads pre-computed images.
    Otherwise uses on-the-fly transformation.
    """
    # Load data once (spectra needed for on-the-fly, labels always needed)
    if config.is_ghs:
        train_spectra, train_labels, test_spectra, test_labels, _ = \
            load_ghs_data(config.data_dir)
    else:
        train_spectra, train_labels, test_spectra, test_labels, _ = \
            load_fold_data(config.data_dir, fold)

    # Fit scaler on train labels
    scaler = fit_scaler(train_labels)
    train_labels_scaled = apply_scaler(train_labels, scaler)
    test_labels_scaled = apply_scaler(test_labels, scaler)

    # Split train into train/val
    idx = np.arange(len(train_labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=config.val_split, random_state=config.seed)

    if cache_dir is not None:
        # Use pre-computed images
        train_data, test_data = load_cached_images(cache_dir, fold)

        if isinstance(train_data, list):
            # Composite: list of memmaps → lazy per-sample stacking
            test_indices = np.arange(len(test_data[0]))
            train_ds = CompositeCachedDataset(
                train_data, train_idx,
                train_labels_scaled[train_idx], augment=True)
            val_ds = CompositeCachedDataset(
                train_data, val_idx,
                train_labels_scaled[val_idx], augment=False)
            test_ds = CompositeCachedDataset(
                test_data, test_indices,
                test_labels_scaled, augment=False)
        else:
            # Single transform cache
            train_ds = CachedDataset(
                train_data[train_idx], train_labels_scaled[train_idx], augment=True)
            val_ds = CachedDataset(
                train_data[val_idx], train_labels_scaled[val_idx], augment=False)
            test_ds = CachedDataset(
                test_data, test_labels_scaled, augment=False)
    else:
        # On-the-fly transform (spectra already loaded above)
        train_ds = OnTheFlyDataset(
            train_spectra[train_idx], train_labels_scaled[train_idx],
            transform, augment=True,
            aug_shift=config.aug_baseline_shift, aug_mult=config.aug_multiplicative)
        val_ds = OnTheFlyDataset(
            train_spectra[val_idx], train_labels_scaled[val_idx],
            transform, augment=False)
        test_ds = OnTheFlyDataset(
            test_spectra, test_labels_scaled, transform, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=True, persistent_workers=config.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers,
                            pin_memory=True, persistent_workers=config.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader, scaler
