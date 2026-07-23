"""Paired significance of our per-trait R^2 vs the reported Cherif et al. 2025 values.

For each of our methods we compute the per-trait delta (ours - Cherif reported) and a
Wilcoxon signed-rank test paired across the 8 traits. Comparisons are always against
Cherif's reported numbers (never against our own methods).

Cherif et al. 2025 reported R^2 (transcribed from the paper, by trait name):
  - Table 2: in-distribution, full-range supervised/self-supervised baselines.
  - Table 4: out-of-distribution cross-dataset baselines.
  Repo: https://github.com/echerif18/HyspectraSSL
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

TRAITS = ['cab', 'cw', 'cm', 'LAI', 'cp', 'cbc', 'car', 'anth']

# Map our metrics.csv trait labels -> canonical names.
_ALIAS = {'Cab': 'cab', 'Car': 'car', 'Anth': 'anth', 'Cw': 'cw', 'Cm': 'cm',
          'LAI': 'LAI', 'Cp': 'cp', 'Cbc': 'cbc'}

# --- Cherif et al. 2025 reported R^2 (by trait name) ------------------------
CHERIF = {
    'table2_id': {  # in-distribution, full-range
        'supervised': {'cab': 0.517, 'cw': 0.621, 'cm': 0.667, 'LAI': 0.561,
                       'cp': 0.651, 'cbc': 0.679, 'car': 0.544, 'anth': 0.454},   # 0.587
        'mae_fr_ft':  {'cab': 0.515, 'cw': 0.634, 'cm': 0.716, 'LAI': 0.615,
                       'cp': 0.676, 'cbc': 0.727, 'car': 0.649, 'anth': 0.598},   # 0.641
    },
    'table4_ood': {  # out-of-distribution cross-dataset
        'supervised': {'cab': 0.362, 'cw': 0.193, 'cm': 0.446, 'LAI': 0.074,
                       'cp': 0.183, 'cbc': 0.449, 'car': 0.181, 'anth': 0.055},   # 0.243
        'mae_fr_ft':  {'cab': 0.271, 'cw': 0.280, 'cm': 0.575, 'LAI': 0.229,
                       'cp': 0.275, 'cbc': 0.582, 'car': 0.165, 'anth': 0.112},   # 0.311
    },
}


def load_ours(results_glob):
    """Mean per-trait R^2 (by canonical name) over seed dirs matching a glob.

    Handles both layouts: <dir>/fold_1/metrics.csv (train_2d) and <dir>/metrics.csv
    (MAE finetune). Averages R^2 across all matching seed directories.
    """
    dirs = sorted(glob.glob(results_glob))
    paths = []
    for d in dirs:
        for cand in (os.path.join(d, 'fold_1', 'metrics.csv'), os.path.join(d, 'metrics.csv')):
            if os.path.isfile(cand):
                paths.append(cand); break
    if not paths:
        raise FileNotFoundError(f"no metrics.csv under {results_glob}")
    frames = [pd.read_csv(p) for p in paths]
    labels = frames[0].iloc[:, 0].tolist()
    r2 = np.stack([f['R2'].values for f in frames]).mean(0)
    return {_ALIAS[l]: v for l, v in zip(labels, r2)}, len(paths)


def delta_wilcoxon(ours, baseline):
    """Per-trait delta (ours - baseline) and Wilcoxon signed-rank across traits."""
    a = np.array([ours[t] for t in TRAITS])
    b = np.array([baseline[t] for t in TRAITS])
    d = a - b
    try:
        w, p = stats.wilcoxon(a, b)
    except ValueError:
        w, p = np.nan, np.nan
    return {'ours': a, 'base': b, 'delta': d, 'ours_avg': a.mean(),
            'base_avg': b.mean(), 'delta_avg': d.mean(), 'wins': int((d > 0).sum()),
            'W': w, 'p': p}


def report(name, ours, baseline_key, table='table2_id'):
    r = delta_wilcoxon(ours, CHERIF[table][baseline_key])
    sig = '*' if (r['p'] == r['p'] and r['p'] < 0.05) else 'n.s.'
    print(f"\n{name}  vs Cherif '{baseline_key}' ({table})")
    print('  ' + '  '.join(f"{t}:{d:+.3f}" for t, d in zip(TRAITS, r['delta'])))
    print(f"  avg ours {r['ours_avg']:.3f}  base {r['base_avg']:.3f}  "
          f"Δ {r['delta_avg']:+.3f}  wins {r['wins']}/8  "
          f"Wilcoxon W={r['W']:.1f} p={r['p']:.3f} {sig}")
    return r


if __name__ == '__main__':
    # Table 1: every transform vs the Cherif supervised 1D baseline (in-distribution).
    print("=" * 70 + "\nTABLE 1 — transforms vs Cherif supervised 1D (ID)\n" + "=" * 70)
    for t in ['reshape', 'serpentine', 'hilbert', 'cwt', 'spectrogram', 'ndi',
              'gaf', 'cos2d', 'mtf']:
        ours, n = load_ours(f'results/greenhs_{t}_efficientnet_b0*')
        report(f'{t} (n={n})', ours, 'supervised', 'table2_id')

    # Table 2 / Table 5: supervised-2D (Reshape) and MAE-2D-FT vs corresponding baselines.
    print("\n" + "=" * 70 + "\nTABLE 2 / TABLE 5 — pretraining strategies vs corresponding Cherif (ID)\n" + "=" * 70)
    resh, _ = load_ours('results/greenhs_reshape_efficientnet_b0*')
    report('Supervised-2D (Reshape)', resh, 'supervised', 'table2_id')
    mae, _ = load_ours('results/mae_finetune_greenhs_s*')
    report('MAE-2D-FT', mae, 'mae_fr_ft', 'table2_id')
