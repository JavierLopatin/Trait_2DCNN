"""Out-of-distribution (cross-dataset) evaluation protocol from Cherif et al. 2025.

This module ports, verbatim in logic, the data-splitting, preprocessing, scaling
and metric code used for the OOD ("cross-dataset generalization", Table 4) setup
of GreenHyperSpectra, so that our 2D-CNN can be evaluated under an identical data
protocol and compared 1:1 to their supervised 1D baseline (avg R^2 = 0.243).

Source (ported with attribution):
    Cherif et al. 2025, GreenHyperSpectra.
    Repo: https://github.com/echerif18/HyspectraSSL  (src/utils_data.py, src/utils_all.py,
          scripts/multi_main_Trans.py)

Only the *data protocol* is reproduced here. The model (Reshape -> EfficientNet-B0)
and the training loop are ours; that 1D-vs-2D difference is exactly what the
comparison measures. The lone code change vs upstream is the modern-pandas fix in
`feature_preparation` (`.fillna(method=...)` -> `.ffill()/.bfill()`), the upstream
version targeting an older pandas.
"""
import random

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import PowerTransformer

# The 8 GHS traits, in the exact order used by Cherif 2025 (multi_main_Trans.py).
GHS_OOD_TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


# --- Spectral preprocessing (Cherif 2025, src/utils_data.py) ----------------

def filter_segment(features_noWtab, order=1, der=False):
    """Savitzky-Golay (window 65) filter of one contiguous spectral segment.
    Ported from Cherif 2025 src/utils_data.py:filter_segment."""
    part1 = features_noWtab.copy()
    if der:
        fr1 = savgol_filter(part1, 65, 1, deriv=1)
    else:
        fr1 = savgol_filter(part1, 65, order)
    return pd.DataFrame(data=fr1, columns=part1.columns)


def feature_preparation(features, inval=(1351, 1431, 1801, 2051), frmax=2451,
                        order=1, der=False):
    """Clip, in-fill and SG-smooth reflectance, removing water bands -> 1721 bands.

    Ported from Cherif 2025 src/utils_data.py:feature_preparation. Only change vs
    upstream: `.fillna(method='ffill'/'bfill')` -> `.ffill()/.bfill()` for modern
    pandas (behaviour identical).
    """
    features = features.copy()
    features.columns = features.columns.astype('int')
    features[features < 0] = 0

    # Substitute high (>1) values with the mean of neighbouring values.
    other = features.copy()
    other[other > 1] = np.nan
    other = (other.ffill(axis=1) + other.bfill(axis=1)) / 2
    other = other.interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1)

    wt_ab = ([i for i in range(inval[0], inval[1])]
             + [i for i in range(inval[2], inval[3])]
             + [i for i in range(2451, 2501)])
    features_noWtab = other.drop(wt_ab, axis=1)

    fr1 = filter_segment(features_noWtab.loc[:, :inval[0] - 1], order=order, der=der)
    fr2 = filter_segment(features_noWtab.loc[:, inval[1]:inval[2] - 1], order=order, der=der)
    fr3 = filter_segment(features_noWtab.loc[:, inval[3]:frmax], order=order, der=der)

    inter = pd.concat([fr1, fr2, fr3], axis=1, join='inner')
    inter[inter < 0] = 0
    return inter


def data_prep_db(db, ls_tr=GHS_OOD_TRAITS):
    """Return (X 1721-band spectra 400-2450, y traits) from a raw labeled frame.
    Ported from Cherif 2025 src/utils_data.py:data_prep_db."""
    X = feature_preparation(db.loc[:, '400':'2500']).loc[:, 400:2450]
    X.index = db.index
    y = db[list(ls_tr)]
    return X, y


def drop_spectral_artifacts(X, y, meta):
    """Drop samples with the red-edge inversion artifact (Cherif 2025
    multi_main_Trans.py): reflectance at 750/1300 nm exceeding that at 1000 nm."""
    red_ed = X.loc[:, 750]
    red_end = X.loc[:, 1300]
    red1000 = X.loc[:, 1000]
    bad = X[(red_end > red1000) & (red_ed > red1000)].index
    if len(bad) > 0:
        X = X.drop(bad)
        y = y.drop(bad)
        meta = meta.drop(bad)
    return X, y, meta


# --- OOD cross-dataset folds (Cherif 2025, src/utils_data.py) ----------------

def sliding_custom_cv(df, seed=42):
    """Yield (df_train, df_test, test_ids) for the cross-dataset OOD folds.

    Verbatim from Cherif 2025 src/utils_data.py:sliding_custom_cv. Fold 0 holds out
    datasets [1,2,4,5,6]; each subsequent fold holds out one of {2,3} plus four
    consecutive datasets, sliding by four. Datasets 2/3 recur across folds by design.
    """
    dataset_ids = sorted(df['dataset'].unique())
    if seed is not None:
        random.seed(seed)

    test_sets = [[1, 2, 4, 5, 6]]
    current = 7
    max_id = max(dataset_ids)
    while current + 3 <= max_id:
        chunk = [current, current + 1, current + 2, current + 3]
        choice = random.choice([2, 3])
        test_sets.append([choice] + chunk)
        current += 4

    for test_ids in test_sets:
        train_ids = [i for i in dataset_ids if i not in test_ids]
        if 2 in train_ids and 3 in train_ids:
            continue
        df_train = df[df['dataset'].isin(train_ids)].copy()
        df_test = df[df['dataset'].isin(test_ids)].copy()
        yield df_train, df_test, test_ids


# --- Box-Cox trait scaler (Cherif 2025, src/utils_data.py:save_scaler) -------

def fit_boxcox_scaler(train_y, standardize=True):
    """Fit a Box-Cox PowerTransformer on the (positive, possibly sparse) traits.
    Matches Cherif 2025 save_scaler(scale=True). sklearn's box-cox ignores NaN in
    fit and preserves NaN in transform, so sparse trait labels are handled directly."""
    return PowerTransformer(method='box-cox', standardize=standardize).fit(np.asarray(train_y))


# --- 1D augmentation (Cherif 2025, src/utils_data.py:SpectraDataset) ---------

class SpectraAugmenter:
    """Per-sample 1D augmentation from Cherif 2025 SpectraDataset: with probability
    `aug_prob`, apply either additive Gaussian noise or a beta/slope/multiplicative
    shift. Ported to operate on numpy spectra (applied before the 2D reshape)."""

    def __init__(self, aug_prob=0.7, betashift=0.01, slopeshift=0.01,
                 multishift=0.1, noise_std=0.01):
        self.aug_prob = aug_prob
        self.betashift = betashift
        self.slopeshift = slopeshift
        self.multishift = multishift
        self.noise_std = noise_std

    def _add_noise(self, x):
        return x + np.random.randn(*x.shape).astype(x.dtype) * self.noise_std

    def _shift(self, x):
        std = x.std()
        beta = (np.random.rand() * 2 * self.betashift - self.betashift) * std
        slope = (np.random.rand() * 2 * self.slopeshift - self.slopeshift + 1)
        axis = np.arange(x.shape[0], dtype=np.float32) / float(x.shape[0])
        offset = (slope * axis + beta - axis - slope / 2.0 + 0.5)
        multi = (np.random.rand() * 2 * self.multishift - self.multishift + 1)
        return np.clip(multi * x + offset * std, 0, None).astype(x.dtype)

    def __call__(self, x):
        if np.random.rand() < self.aug_prob:
            return random.choice([self._add_noise, self._shift])(x)
        return x


# --- Masked Huber loss / NaN handling (Cherif 2025, src/utils_all.py) --------

import torch
import torch.nn as nn


class HuberCustomLoss(nn.Module):
    """Masked Huber loss, ported verbatim from Cherif 2025 src/utils_all.py.

    This *is* Cherif's NaN handling for training: only finite (observed) entries of
    both target and prediction contribute (``finite_mask``), so the sparse/NaN trait
    labels are masked out exactly as in their pipeline. Labels are expected already in
    Box-Cox space (their transformation_layer is applied before the loss); NaN labels
    stay NaN through Box-Cox and are dropped here. Huber transitions MSE->MAE at
    ``threshold`` (=1.0 in the baseline). Without sample weights, returns the mean over
    finite entries flattened across traits and samples.
    """

    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, y_true, y_pred, sample_weight=None):
        y_true = y_true.to(y_pred.device)
        finite_mask = torch.isfinite(y_pred) & torch.isfinite(y_true)
        error = y_pred[finite_mask] - y_true[finite_mask]
        abs_error = torch.abs(error)
        squared_loss = 0.5 * error ** 2
        linear_loss = self.threshold * abs_error - 0.5 * self.threshold ** 2
        loss = torch.where(abs_error < self.threshold, squared_loss, linear_loss)
        if sample_weight is not None:
            sw = torch.stack([sample_weight for _ in range(y_true.size(1))], dim=1).to(y_pred.device)
            return (loss * sw[finite_mask]).sum()
        return loss.mean()


# --- Metrics (Cherif 2025, src/utils_all.py:eval_metrics) --------------------

def eval_metrics(obs, pred):
    """Per-trait R2, RMSE, nRMSE(%), MAE, Bias, RPD, Spearman^2 for observed vs
    predicted trait frames (columns = traits). Ported from Cherif 2025 eval_metrics.
    nRMSE is normalized by the 1-99% quantile range of the observations."""
    import math
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    def rpd(y_true, y_pred):
        return np.std(y_true, ddof=1) / np.sqrt(mean_squared_error(y_true, y_pred))

    rows = {k: [] for k in ['r2_score', 'RMSE', 'nRMSE (%)', 'MAE', 'Bias', 'RPD',
                            'spearmanr_squared']}
    traits = list(obs.columns)
    for j in traits:
        f = pred[j].reset_index(drop=True)
        y = obs[j].reset_index(drop=True)
        idx = np.union1d(f[f.isna()].index, y[y.isna()].index)
        f = f.drop(idx).reset_index(drop=True)
        y = y.drop(idx).reset_index(drop=True)
        if y.notnull().sum():
            rmse = math.sqrt(mean_squared_error(y, f))
            rng = np.nanquantile(np.array(y), 0.99) - np.nanquantile(np.array(y), 0.01)
            rows['r2_score'].append(r2_score(y, f))
            rows['RMSE'].append(rmse)
            rows['nRMSE (%)'].append((rmse * 100) / rng)
            rows['MAE'].append(mean_absolute_error(y, f))
            rows['Bias'].append(np.sum(np.array(y) - np.array(f)) / len(f))
            rows['RPD'].append(rpd(y, f))
            rows['spearmanr_squared'].append(stats.spearmanr(y, f).statistic ** 2)
        else:
            for k in rows:
                rows[k].append(np.nan)
    return pd.DataFrame(rows, index=traits)


def subsample_macro_metrics(obs, pred, dataset_ids, n_sub=5, max_n=30, seed=0):
    """Macro metric over `n_sub` random subsamples, each capped at `max_n` samples
    per held-out dataset (Cherif 2025 OOD evaluation, §5). Reduces R^2 sensitivity to
    dataset imbalance. Returns (mean_df, std_df) of per-trait metrics over the subsamples.

    obs/pred: DataFrames (rows = pooled held-out samples, cols = traits), same index.
    dataset_ids: array-like of the source dataset id per row (aligned to obs/pred).
    """
    obs = obs.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    ds = np.asarray(dataset_ids)
    rng = np.random.RandomState(seed)
    per_sub = []
    for _ in range(n_sub):
        take = []
        for d in np.unique(ds):
            rows = np.where(ds == d)[0]
            k = min(max_n, len(rows))
            take.append(rng.choice(rows, size=k, replace=False))
        sel = np.concatenate(take)
        per_sub.append(eval_metrics(obs.iloc[sel], pred.iloc[sel]))
    stacked = np.stack([m.values for m in per_sub])          # (n_sub, traits, metrics)
    cols = per_sub[0].columns
    idx = per_sub[0].index
    mean_df = pd.DataFrame(np.nanmean(stacked, 0), index=idx, columns=cols)
    std_df = pd.DataFrame(np.nanstd(stacked, 0), index=idx, columns=cols)
    return mean_df, std_df
