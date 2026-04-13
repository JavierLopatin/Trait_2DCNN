"""Post-process PROSAIL LUT to match GreenHyperSpectra spectral format.

Steps:
    1. Clip reflectance values (< 0 → 0, > 1 → interpolate)
    2. Savitzky-Golay smoothing (window=65, poly=1) per segment
    3. Remove water absorption bands (1351-1430, 1801-2050, 2451-2500)
    4. Add multiplicative Gaussian noise (2% std)
    5. Export with columns matching GreenHyperSpectra format (1721 bands)

Usage:
    python prosail/postprocess_lut.py [--input raw_lut.csv] [--output processed.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from training.config import GHS_TRAIT_NAMES


# Water absorption bands to remove (same as Cherif preprocessing)
WATER_BANDS = [
    (1351, 1430),
    (1801, 2050),
    (2451, 2500),
]


def clip_reflectance(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Clip invalid reflectance and interpolate values > 1.

    Negative values are clipped to 0. Values > 1 are replaced via linear
    interpolation from neighboring valid bands. Only rows containing at
    least one value > 1 enter the per-sample interpolation loop.
    """
    spectra = spectra.copy()
    spectra[spectra < 0] = 0

    # Only iterate over samples that actually need interpolation
    over_mask = spectra > 1
    rows_needing_interp = np.where(over_mask.any(axis=1))[0]

    if len(rows_needing_interp) > 0:
        spectra[over_mask] = np.nan
        for i in rows_needing_interp:
            nan_mask = np.isnan(spectra[i])
            spectra[i, nan_mask] = np.interp(
                wavelengths[nan_mask], wavelengths[~nan_mask],
                spectra[i, ~nan_mask])
    return spectra


def smooth_segments(spectra: np.ndarray, wavelengths: np.ndarray,
                    window: int = 65, polyorder: int = 1) -> np.ndarray:
    """Apply Savitzky-Golay smoothing per spectral segment.

    Segments are defined by water absorption gaps:
    - Segment 1: 400-1350 nm
    - Segment 2: 1431-1800 nm
    - Segment 3: 2051-2450 nm
    """
    segments = [
        (400, 1350),
        (1431, 1800),
        (2051, 2450),
    ]
    smoothed = spectra.copy()
    for start, end in segments:
        mask = (wavelengths >= start) & (wavelengths <= end)
        n_bands = mask.sum()
        if n_bands < window:
            win = min(n_bands, window)
            if win % 2 == 0:
                win -= 1
            if win < 3:
                continue
        else:
            win = window
        smoothed[:, mask] = savgol_filter(
            spectra[:, mask], window_length=win, polyorder=polyorder, axis=1)
    return smoothed


def remove_water_bands(spectra: np.ndarray, wavelengths: np.ndarray):
    """Remove water absorption bands, returning 1721 bands."""
    keep = np.ones(len(wavelengths), dtype=bool)
    for start, end in WATER_BANDS:
        keep &= ~((wavelengths >= start) & (wavelengths <= end))
    return spectra[:, keep], wavelengths[keep]


def add_noise(spectra: np.ndarray, noise_std: float = 0.02,
              seed: int = 42) -> np.ndarray:
    """Add multiplicative Gaussian noise (Ferreira et al. 2025 style)."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, spectra.shape)
    return spectra * (1 + noise)


def postprocess(input_path: str, output_path: str, noise_std: float = 0.02):
    """Full post-processing pipeline."""
    print(f"Loading raw LUT: {input_path}")
    df = pd.read_csv(input_path)

    # Separate traits and spectra
    trait_cols = [c for c in GHS_TRAIT_NAMES if c in df.columns]
    spec_cols = [c for c in df.columns if c not in trait_cols]
    wavelengths = np.array([float(c) for c in spec_cols])
    spectra = df[spec_cols].values.astype(np.float64)
    traits = df[trait_cols]

    print(f"  Samples: {len(spectra)}")
    print(f"  Input bands: {len(wavelengths)} ({wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm)")

    # Step 1: Clip
    print("Step 1: Clipping reflectance...")
    spectra = clip_reflectance(spectra, wavelengths)

    # Step 2: Savitzky-Golay smoothing
    print("Step 2: Savitzky-Golay smoothing (window=65, poly=1)...")
    spectra = smooth_segments(spectra, wavelengths, window=65, polyorder=1)

    # Step 3: Remove water bands
    print("Step 3: Removing water absorption bands...")
    spectra, wavelengths_clean = remove_water_bands(spectra, wavelengths)
    print(f"  Output bands: {len(wavelengths_clean)} ({wavelengths_clean[0]:.0f}-{wavelengths_clean[-1]:.0f} nm)")

    # Step 4: Add noise
    print(f"Step 4: Adding multiplicative Gaussian noise (std={noise_std})...")
    spectra = add_noise(spectra, noise_std=noise_std)
    # Clip again after noise
    spectra = np.clip(spectra, 0, None)

    # Build output dataframe
    spec_df = pd.DataFrame(spectra, columns=[str(int(w)) for w in wavelengths_clean])
    output = pd.concat([traits.reset_index(drop=True), spec_df], axis=1)

    print(f"\nSaving to {output_path}")
    output.to_csv(output_path, index=False)
    print(f"  Final: {len(output)} samples x {output.shape[1]} columns "
          f"({len(trait_cols)} traits + {len(wavelengths_clean)} bands)")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Post-process PROSAIL LUT')
    parser.add_argument('--input', type=str,
                        default='data/GreenHyperSpectra/prosail_lut_raw.csv')
    parser.add_argument('--output', type=str,
                        default='data/GreenHyperSpectra/prosail_lut.csv')
    parser.add_argument('--noise-std', type=float, default=0.02,
                        help='Multiplicative noise std (default: 0.02 = 2%%)')
    args = parser.parse_args()

    postprocess(args.input, args.output, args.noise_std)


if __name__ == '__main__':
    main()
