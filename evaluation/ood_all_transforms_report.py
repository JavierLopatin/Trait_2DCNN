"""Aggregate the OOD (Cherif 2025 protocol) cross-dataset results across every
1D->2D transform and emit a Markdown report.

For each transform we read results/ood_<name>/ood_metrics_mean.csv (+ _std) produced
by training/train_ood.py, compute the per-trait and average R^2 / nRMSE, the per-trait
delta vs the Cherif et al. 2025 supervised 1D OOD baseline (Table 4), and a Wilcoxon
signed-rank test paired across the 8 traits. Comparisons are always against Cherif's
reported numbers, never against our own methods.

Usage:
    python -m evaluation.ood_all_transforms_report
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

TRAITS = ['cab', 'cw', 'cm', 'LAI', 'cp', 'cbc', 'car', 'anth']

# Cherif et al. 2025, Table 4 (out-of-distribution cross-dataset), reported R^2.
CHERIF_SUP_1D = {'cab': 0.362, 'cw': 0.193, 'cm': 0.446, 'LAI': 0.074,
                 'cp': 0.183, 'cbc': 0.449, 'car': 0.181, 'anth': 0.055}   # avg 0.243
CHERIF_MAE_1D = {'cab': 0.271, 'cw': 0.280, 'cm': 0.575, 'LAI': 0.229,
                 'cp': 0.275, 'cbc': 0.582, 'car': 0.165, 'anth': 0.112}   # avg 0.311

# Every transform we evaluate OOD (results dir name -> display label).
TRANSFORMS = [
    ('reshape', 'Reshape'), ('serpentine', 'Serpentine'), ('hilbert', 'Hilbert'),
    ('mtf', 'MTF'), ('cwt', 'CWT'), ('cos2d', 'coCorr-2D'), ('gaf', 'GAF'),
    ('ndi', 'NDI (3ch)'), ('spectrogram', 'Spectrogram'),
    ('reshape+cwt+ndi', 'Reshape+CWT+NDI (5ch)'),
]


def load_mean_std(name):
    d = f'results/ood_{name}'
    mp, sp = f'{d}/ood_metrics_mean.csv', f'{d}/ood_metrics_std.csv'
    if not os.path.isfile(mp):
        return None, None
    mean = pd.read_csv(mp, index_col=0)
    std = pd.read_csv(sp, index_col=0) if os.path.isfile(sp) else None
    return mean, std


def wilcoxon_vs(r2_by_trait, baseline):
    a = np.array([r2_by_trait[t] for t in TRAITS])
    b = np.array([baseline[t] for t in TRAITS])
    d = a - b
    try:
        w, p = stats.wilcoxon(a, b)
    except ValueError:
        w, p = np.nan, np.nan
    return d, w, p, int((d > 0).sum())


def main():
    rows = []
    for name, label in TRANSFORMS:
        mean, std = load_mean_std(name)
        if mean is None:
            print(f"  [skip] {name}: no results yet")
            continue
        r2 = {t: float(mean.loc[t, 'r2_score']) for t in TRAITS}
        nrmse = float(mean.loc[:, 'nRMSE (%)'].mean())
        avg = float(np.mean([r2[t] for t in TRAITS]))
        d, w, p, wins = wilcoxon_vs(r2, CHERIF_SUP_1D)
        rows.append(dict(name=name, label=label, r2=r2, avg=avg, nrmse=nrmse,
                         delta=d, W=w, p=p, wins=wins))

    rows.sort(key=lambda r: r['avg'], reverse=True)

    # ---- Markdown ----------------------------------------------------------
    md = []
    md.append("# OOD cross-dataset — all 1D→2D transforms vs Cherif et al. 2025\n")
    md.append("Out-of-distribution (cross-dataset) evaluation replicating the Cherif "
              "et al. 2025 GreenHyperSpectra protocol **1:1** (12 sliding folds, "
              "`feature_preparation`, red-edge artifact filter, internal 80/20 stratified "
              "split, Box-Cox scaling, masked-Huber loss, sub-sampled macro metric "
              "≤30/dataset ×5). Only the input representation changes: 1D spectrum → "
              "each transform → EfficientNet-B0. 1 seed. Baseline = Cherif supervised 1D "
              "(avg R² **0.243**); their best OOD model MAE-FR-FT 1D = **0.311**.\n")

    # Summary table (avg R2 sorted, delta vs supervised 1D, Wilcoxon)
    md.append("## Average across 8 traits\n")
    md.append("| Rank | Transform | avg R² | ΔR² vs 1D-sup | wins/8 | Wilcoxon p | nRMSE (%) |")
    md.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        sig = '' if not (r['p'] == r['p']) else (' *' if r['p'] < 0.05 else '')
        dv = r['avg'] - 0.243
        md.append(f"| {i} | {r['label']} | {r['avg']:.3f} | {dv:+.3f} | "
                  f"{r['wins']}/8 | {r['p']:.3f}{sig} | {r['nrmse']:.2f} |")
    md.append("\n_Cherif supervised 1D avg R² = 0.243; MAE-FR-FT 1D = 0.311. "
              "`*` = Wilcoxon p<0.05 across the 8 traits._\n")

    # Per-trait R2 table
    md.append("## Per-trait R²\n")
    md.append("| Transform | " + " | ".join(TRAITS) + " | avg |")
    md.append("|" + "---|" * (len(TRAITS) + 2))
    md.append("| _Cherif 1D-sup_ | " + " | ".join(f"{CHERIF_SUP_1D[t]:.3f}" for t in TRAITS)
              + " | 0.243 |")
    for r in rows:
        md.append(f"| {r['label']} | "
                  + " | ".join(f"{r['r2'][t]:.3f}" for t in TRAITS)
                  + f" | {r['avg']:.3f} |")
    md.append("")

    out = 'paper/ood_all_transforms_analysis.md'
    with open(out, 'w') as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\n[written] {out}")


if __name__ == '__main__':
    main()
