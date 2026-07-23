"""Paired significance tests for the OOD cross-dataset comparison.

Compares our OOD per-trait R^2 (from training/train_ood.py outputs) against the
published Cherif et al. 2025 baselines (GreenHyperSpectra, Table 4) with a paired
test across the 8 traits -- the same paired-across-traits approach used for the
in-distribution transform comparison (Case Study 1).

Only a paired-across-traits test is available: Cherif 2025 report per-trait point
estimates (Table 4), not per-sample predictions, so a per-sample paired test against
their models is not possible. Statistical power is inherently limited (n = 8 traits,
single training run per the OOD protocol).

Baselines transcribed from Cherif et al. 2025, Table 4 (OOD cross-dataset R^2):
    https://github.com/echerif18/HyspectraSSL

Usage:
    python -m evaluation.ood_significance --ours results/ood_reshape --baseline supervised
    python -m evaluation.ood_significance --ours results/ood_mae_ft --baseline mae_fr_ft
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Trait order used throughout (Cherif 2025 Table 4 column order).
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]

# Cherif et al. 2025, Table 4 -- OOD cross-dataset R^2 per trait.
CHERIF2025_TABLE4_R2 = {
    "supervised": [0.362, 0.193, 0.446, 0.074, 0.183, 0.449, 0.181, 0.055],  # avg 0.243
    "sr_gan":     [0.300, 0.350, 0.507, -0.199, 0.273, 0.548, 0.221, 0.197],  # avg 0.275
    "rtm_ae":     [0.272, 0.193, 0.453, 0.019, 0.192, -0.075, 0.266, 0.067],  # avg 0.173
    "mae_fr_lp":  [0.116, 0.298, 0.442, 0.182, 0.211, 0.478, 0.232, 0.142],   # avg 0.263
    "mae_fr_ft":  [0.271, 0.280, 0.575, 0.229, 0.275, 0.582, 0.165, 0.112],   # avg 0.311
}


def load_our_r2(results_dir):
    """Per-trait OOD R^2 (in TRAITS order) from an ood_metrics_mean.csv."""
    df = pd.read_csv(Path(results_dir) / 'ood_metrics_mean.csv', index_col=0)
    return np.array([df.loc[t, 'r2_score'] for t in TRAITS])


def paired_test(ours, baseline_name):
    """Paired t-test and Wilcoxon signed-rank of our per-trait R^2 vs a Cherif
    baseline, across the 8 traits. Returns a dict of results."""
    base = np.array(CHERIF2025_TABLE4_R2[baseline_name])
    d = ours - base
    t_stat, p_t = stats.ttest_rel(ours, base)
    # Wilcoxon needs >0 non-zero differences; guard tiny samples.
    try:
        w_stat, p_w = stats.wilcoxon(ours, base)
    except ValueError:
        w_stat, p_w = np.nan, np.nan
    return {
        'baseline': baseline_name,
        'ours_avg': float(ours.mean()),
        'base_avg': float(base.mean()),
        'delta_avg': float(d.mean()),
        'wins': int((d > 0).sum()),
        't': float(t_stat), 'p_ttest': float(p_t),
        'W': float(w_stat), 'p_wilcoxon': float(p_w),
        'per_trait_delta': dict(zip(TRAITS, d.round(3))),
    }


def print_report(ours, baseline_name):
    base = np.array(CHERIF2025_TABLE4_R2[baseline_name])
    r = paired_test(ours, baseline_name)
    print(f"\n=== OOD paired test across 8 traits: ours vs Cherif '{baseline_name}' ===")
    print(f"{'trait':6s}{'ours':>8s}{'base':>8s}{'delta':>8s}")
    for t, o, b in zip(TRAITS, ours, base):
        print(f"{t:6s}{o:8.3f}{b:8.3f}{o-b:+8.3f}")
    print(f"{'AVG':6s}{r['ours_avg']:8.3f}{r['base_avg']:8.3f}{r['delta_avg']:+8.3f}"
          f"   (wins {r['wins']}/8)")
    sig = '*' if r['p_wilcoxon'] < 0.05 else 'n.s.'
    print(f"  Wilcoxon signed-rank (paired across traits): "
          f"W={r['W']:.1f}  p={r['p_wilcoxon']:.3f}  {sig}")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ours', required=True, help='OOD results dir (has ood_metrics_mean.csv)')
    ap.add_argument('--baseline', default='supervised', choices=list(CHERIF2025_TABLE4_R2))
    args = ap.parse_args()
    print_report(load_our_r2(args.ours), args.baseline)


if __name__ == '__main__':
    main()
