"""Turn the PROSAIL trait-sweep LUT into per-band theoretical sensitivity.

Reuses the LUT post-processing (per-segment Savitzky-Golay smoothing + water-band
removal) so the theoretical curves live on exactly the same 1721-band axis as the
model input. No noise is added: these are clean theoretical reference curves.

Sensitivity S_trait(lambda) = coefficient of variation (std / mean), across the
trait's swept reflectance curves, at each band -- i.e. the *relative* strength of
that band's response when the trait varies and everything else is held fixed.
Normalized to [0, 1] per trait.

The relative (CV) form rather than the raw std deliberately removes the trivial
"high absolute reflectance -> high absolute variation" effect: raw std peaks on the
bright NIR plateau (~1074 nm) for every constituent, the same absolute-magnitude
artifact that motivated the B-inv choice in the Integrated-Gradients analysis. CV
highlights the diagnostic absorption features instead, and empirically it removes
the spurious *negative* IG-vs-theory correlations that raw std produces for the
dry-matter traits (Cm, Cbc).

Usage:
    python prosail/postprocess_sensitivity.py
    python prosail/postprocess_sensitivity.py --input ... --output ...
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prosail.postprocess_lut import smooth_segments, remove_water_bands

try:
    from training.config import GHS_TRAIT_SHORT
except ModuleNotFoundError:
    GHS_TRAIT_SHORT = ['Cab', 'Car', 'Anth', 'Cw', 'Cm', 'LAI', 'Cp', 'Cbc']

META_COLS = ['trait', 'param', 'param_value']


def postprocess(input_path: str, output_path: str):
    print(f"Loading sweep LUT: {input_path}")
    df = pd.read_csv(input_path)

    spec_cols = [c for c in df.columns if c not in META_COLS]
    wavelengths = np.array([float(c) for c in spec_cols])
    print(f"  Rows: {len(df)}  Input bands: {len(wavelengths)} "
          f"({wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm)")

    spectra = df[spec_cols].values.astype(np.float64)

    # Same smoothing + water-band removal as the training LUT (no noise).
    print("Smoothing (Savitzky-Golay, per segment) + removing water bands...")
    spectra = smooth_segments(spectra, wavelengths, window=65, polyorder=1)
    spectra, wl_clean = remove_water_bands(spectra, wavelengths)
    print(f"  Output bands: {len(wl_clean)} "
          f"({wl_clean[0]:.0f}-{wl_clean[-1]:.0f} nm)")

    # Per-trait sensitivity = coefficient of variation (std/mean) across swept
    # curves at each band, normalized to [0, 1]. The mean floor guards deep
    # absorption bands where reflectance approaches zero.
    out = {'wavelength_nm': wl_clean.astype(int)}
    for trait in GHS_TRAIT_SHORT:
        mask = (df['trait'] == trait).values
        curves = spectra[mask]                       # (n_steps, n_bands)
        sens = curves.std(axis=0) / np.clip(curves.mean(axis=0), 1e-3, None)
        mx = sens.max()
        out[trait] = sens / mx if mx > 0 else sens
        peak_nm = int(wl_clean[np.argmax(sens)])
        print(f"  {trait:4s}: {mask.sum():2d} curves, peak sensitivity @ {peak_nm} nm")

    out_df = pd.DataFrame(out)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} "
          f"({len(out_df)} bands x {len(GHS_TRAIT_SHORT)} traits)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=str,
                    default='data/GreenHyperSpectra/prosail_sensitivity_raw.csv')
    ap.add_argument('--output', type=str,
                    default='results/interpretation/prosail_sensitivity_per_band.csv')
    args = ap.parse_args()
    postprocess(args.input, args.output)


if __name__ == '__main__':
    main()
