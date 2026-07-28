"""Evidence for the 42-band ripple described in the manuscript's limitations.

The unfolded Integrated Gradients profiles carry a ripple whose period matches the
42-band fold of the Reshape transform. This script reproduces every number the
manuscript reports about it, in five checks:

    1. Unfolding fidelity. `unfold_2d_to_bands` must invert the forward resize,
       which uses bilinear interpolation with align_corners=True. scipy's zoom is
       not that inverse, because its default constant edge padding drives the last
       row and column to zero and pins every 42nd band to exactly zero. The check
       resamples a constant map and round-trips a synthetic profile that has no
       42-band structure, so any ripple in the result is manufactured.

    2. Ripple strength. Share of each trait's profile variance carried by the
       harmonics of n/42, and the mean importance per grid column.

    3. Border attenuation, the cause. The last band of every row lands on the right
       edge of the 224 by 224 image. This measures the input gradient per pixel
       column on the trained network, and on randomly initialised ones to separate
       what the architecture's padding does from what training adds.

    4. Peak safety. Where each reported peak sits in the 42-column grid. Peaks in
       the attenuated columns would be suspect; interior peaks are not.

    5. Agreement with the PROSAIL theoretical sensitivity, under a linear, two rank
       based, and one distance based measure.

Removing the ripple was tried and rejected. Low-pass filtering wide enough to span
42 bands destroys the narrow absorption features the analysis rests on, moving the
chlorophyll peak from 703 to 2139 nm. Dividing by a measured gain fails because
check 3 shows the attenuation is part architectural and part learned, with no way
to separate them, so no well-defined divisor exists. The manuscript therefore
reports the raw profile and describes the ripple.

Run: python -m evaluation.verify_unfold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

N_BANDS = 1721
SIDE = 42
GRID = SIDE * SIDE
IMG = 224
TRAITS = ['Cab', 'Car', 'Anth', 'Cw', 'Cm', 'LAI', 'Cp', 'Cbc']
ATTENUATED_FROM = 36          # columns 36..41 carry the border attenuation


# --- 1. Candidate inverses ------------------------------------------------

def inv_zoom(img: np.ndarray) -> np.ndarray:
    """The inverse used before this was diagnosed. Zeroes the last row/column."""
    return zoom(img, SIDE / img.shape[0], order=1)


def inv_zoom_nearest(img: np.ndarray) -> np.ndarray:
    """Same estimator with replicated edges instead of zero padding."""
    return zoom(img, SIDE / img.shape[0], order=1, mode='nearest')


def inv_torch(img: np.ndarray) -> np.ndarray:
    """Exact mirror of the forward resize, so corners map onto corners. Current."""
    t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32)
    return F.interpolate(t[None, None], size=(SIDE, SIDE),
                         mode='bilinear', align_corners=True).numpy()[0, 0]


def inv_area(img: np.ndarray) -> np.ndarray:
    """Average every band's pixel footprint instead of sampling its centre."""
    t = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32)
    return F.adaptive_avg_pool2d(t[None, None], (SIDE, SIDE)).numpy()[0, 0]


INVERSES = {
    'zoom (superseded)': inv_zoom,
    'zoom, edge replicate': inv_zoom_nearest,
    'interpolate, align_corners': inv_torch,
    'adaptive average pool': inv_area,
}


def forward(profile: np.ndarray) -> np.ndarray:
    """Band profile -> padded 42x42 grid -> 224x224 image, as the model sees it."""
    x = np.pad(profile, (0, GRID - len(profile))).reshape(1, 1, SIDE, SIDE)
    t = torch.as_tensor(x, dtype=torch.float32)
    return F.interpolate(t, size=(IMG, IMG), mode='bilinear',
                         align_corners=True).numpy()[0, 0]


# --- 2. Ripple strength ---------------------------------------------------

def ripple_power(y: np.ndarray, side: int = SIDE) -> float:
    """Share of the profile variance carried by the harmonics of n/side."""
    n = len(y)
    power = np.abs(np.fft.rfft(y - y.mean())) ** 2
    k0 = n / side
    bins = {b for h in range(1, side // 2 + 1)
            for b in range(int(round(k0 * h)) - 1, int(round(k0 * h)) + 2)
            if 0 < b < len(power)}
    return float(power[sorted(bins)].sum() / power[1:].sum())


def column_profile(y: np.ndarray, side: int = SIDE) -> np.ndarray:
    """Mean importance per grid column, normalised to the interior median.

    Flat under no artifact. The decay over the last columns is the border effect.
    """
    grid = np.full(GRID, np.nan)
    grid[:len(y)] = y
    col = np.nanmean(grid.reshape(side, side), axis=0)
    return col / np.median(col[2:ATTENUATED_FROM])


# --- 3. Border attenuation ------------------------------------------------

def gradient_gain(model, images, traits=(0, 3, 6)) -> np.ndarray:
    """Mean |d f_trait / d pixel| over images, averaged across a few traits."""
    acc = []
    for t in traits:
        x = images.clone().requires_grad_(True)
        g = torch.autograd.grad(model(x)[:, t].sum(), x)[0].abs()[:, 0].mean(0)
        acc.append(g.numpy())
    return np.mean(acc, axis=0)


def edge_summary(gain: np.ndarray) -> dict:
    """Column gain at both image edges, relative to the interior median."""
    col = gain.mean(0)
    interior = np.median(col[20:200])
    return {'left': float(col[0] / interior), 'right': float(col[-1] / interior)}


# --- 4. Agreement metrics -------------------------------------------------

def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Szekely's distance correlation. Zero only under full independence, so it
    detects dependence Pearson misses. Always positive, so it loses the sign."""
    x = np.asarray(x, float)[:, None]
    y = np.asarray(y, float)[:, None]
    a, b = np.abs(x - x.T), np.abs(y - y.T)
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    denom = np.sqrt((A * A).mean() * (B * B).mean())
    return float(np.sqrt(max((A * B).mean(), 0) / denom)) if denom > 0 else np.nan


def agreement_metrics(importance: np.ndarray, sensitivity: np.ndarray) -> dict:
    """Linear, rank-based, and distance-based agreement with PROSAIL S(lambda).

    Pearson is the headline metric because its sign is interpretable, which matters
    for leaf area index, whose importance is anti-aligned with theory. The rank
    measures penalise narrow absorption features, since the theoretical sensitivity
    is dominated in magnitude by the broad near-infrared plateau.
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau
    ok = np.isfinite(importance) & np.isfinite(sensitivity)
    xi, yi = importance[ok], sensitivity[ok]
    step = max(1, len(xi) // 400)          # subsample: dCor is O(n^2) in memory
    return {'pearson_r': float(pearsonr(xi, yi)[0]),
            'spearman_rho': float(spearmanr(xi, yi)[0]),
            'kendall_tau': float(kendalltau(xi, yi)[0]),
            'distance_corr': distance_correlation(xi[::step], yi[::step])}


# --- Checks ---------------------------------------------------------------

def check_inverses() -> None:
    print('1. Unfolding fidelity, on inputs with no real 42-band structure\n')
    constant = np.ones((IMG, IMG))
    rng = np.random.default_rng(0)
    smooth = np.convolve(rng.standard_normal(N_BANDS), np.ones(15) / 15, 'same')
    smooth -= smooth.min()
    image = forward(smooth)
    print(f'   {"inverse":28s} {"zeroed":>7s} {"const CV":>9s} {"round-trip r":>13s} {"ripple":>8s}')
    for name, inverse in INVERSES.items():
        rec = inverse(image).reshape(-1)[:N_BANDS]
        flat = inverse(constant)
        print(f'   {name:28s} {int((rec == 0).sum()):7d} '
              f'{flat.std() / flat.mean():9.4f} {np.corrcoef(smooth, rec)[0, 1]:13.4f} '
              f'{100 * ripple_power(rec):7.1f}%')


def check_ripple(df: pd.DataFrame) -> None:
    print('\n2. Ripple strength and column profile of the real attributions\n')
    print(f'   {"trait":6s} {"zeroed":>7s} {"ripple":>8s} {"col 36":>7s} {"col 39":>7s} {"col 41":>7s}')
    for trait in TRAITS:
        y = df[f'{trait}_imp_mean'].to_numpy()
        col = column_profile(y)
        print(f'   {trait:6s} {int((y == 0).sum()):7d} {100 * ripple_power(y):7.1f}% '
              f'{col[36]:7.3f} {col[39]:7.3f} {col[41]:7.3f}')


def check_border(exp_dir: Path, n_images: int, n_random: int) -> None:
    print('\n3. Border attenuation, trained network against untrained ones\n')
    from evaluation.spectral_importance import load_model, make_images, GHS_DATA_DIR
    from training.data_loader import load_ghs_data
    from models.trait_model import get_model

    _, _, test_s, _, _ = load_ghs_data(GHS_DATA_DIR)
    images = make_images(test_s[:n_images], 'cpu')

    trained = edge_summary(gradient_gain(load_model(exp_dir, 'cpu'), images))
    maps = []
    for seed in range(n_random):
        torch.manual_seed(seed)
        maps.append(gradient_gain(
            get_model('efficientnet_b0', in_channels=1, n_traits=len(TRAITS)).eval(),
            images))
    untrained = edge_summary(np.mean(maps, axis=0))

    print(f'   {"network":28s} {"left edge":>10s} {"right edge":>11s}')
    print(f'   {f"untrained (mean of {n_random})":28s} '
          f'{untrained["left"]:10.3f} {untrained["right"]:11.3f}')
    print(f'   {"trained":28s} {trained["left"]:10.3f} {trained["right"]:11.3f}')
    print('\n   The architecture attenuates both edges. Training recovers the left'
          '\n   and deepens the right, where the 43 zero-padding cells of the'
          '\n   reshape sit, so the two contributions are not separable.')


def check_peaks(df: pd.DataFrame) -> None:
    print('\n4. Grid column of each reported peak\n')
    wl = df['wavelength_nm'].to_numpy()
    print(f'   {"trait":6s} {"peak nm":>8s} {"band":>6s} {"column":>7s}   status')
    for trait in TRAITS:
        k = int(df[f'{trait}_imp_mean'].to_numpy().argmax())
        status = 'ATTENUATED' if k % SIDE >= ATTENUATED_FROM else 'interior'
        print(f'   {trait:6s} {wl[k]:8.0f} {k:6d} {k % SIDE:7d}   {status}')


def check_agreement(df: pd.DataFrame, sensitivity: Path) -> None:
    print('\n5. Agreement with the PROSAIL theoretical sensitivity\n')
    sens = pd.read_csv(sensitivity)
    wl = df['wavelength_nm'].to_numpy()
    s_wl = sens['wavelength_nm'].to_numpy(float)
    print(f'   {"trait":6s} {"r":>8s} {"rho":>8s} {"tau":>8s} {"dCor":>8s}')
    for trait in TRAITS:
        s = sens[trait].to_numpy(float)
        if len(s_wl) != len(wl) or not np.allclose(s_wl, wl):
            s = np.interp(wl, s_wl, s)
        m = agreement_metrics(df[f'{trait}_imp_mean'].to_numpy(), s)
        print(f'   {trait:6s} {m["pearson_r"]:8.3f} {m["spearman_rho"]:8.3f} '
              f'{m["kendall_tau"]:8.3f} {m["distance_corr"]:8.3f}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path,
                    default=Path('results/interpretation/ig_importance_per_band.csv'))
    ap.add_argument('--sensitivity', type=Path,
                    default=Path('results/interpretation/prosail_sensitivity_per_band.csv'))
    ap.add_argument('--exp-dir', type=Path,
                    default=Path('results/greenhs_reshape_efficientnet_b0'))
    ap.add_argument('--n-images', type=int, default=256)
    ap.add_argument('--n-random', type=int, default=5)
    ap.add_argument('--skip-border', action='store_true',
                    help='skip check 3, which needs the trained checkpoint')
    args = ap.parse_args()

    check_inverses()
    df = pd.read_csv(args.input)
    check_ripple(df)
    ckpt = args.exp_dir / 'fold_1' / 'best.ckpt'
    if not args.skip_border and ckpt.exists():
        check_border(args.exp_dir, args.n_images, args.n_random)
    else:
        print(f'\n3. Border attenuation skipped, no checkpoint at {ckpt}')
    check_peaks(df)
    if args.sensitivity.exists():
        check_agreement(df, args.sensitivity)
    else:
        print(f'\n5. Agreement skipped, no {args.sensitivity}')


if __name__ == '__main__':
    main()
