"""Compare 2D methods against Cherif et al. baselines.

Usage:
    python -m evaluation.compare_methods --results-dir results/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from training.config import TRAIT_SHORT


def load_cherif_baselines(models_dir: Path = Path('multi-traitretrieval/models')):
    """Load Cherif 1D-CNN baseline results from stored predictions."""
    results = {}

    for exp_type in ['multi', 'plsr_sing']:
        all_r2 = []
        for fold in range(1, 6):
            pred_path = models_dir / f'{exp_type}_{fold}'
            if exp_type == 'multi':
                pred_path = pred_path / 'inexact'

            pred_file = pred_path / 'Predictions.csv'
            obs_file = pred_path / 'Observations.csv'

            if not pred_file.exists() or not obs_file.exists():
                continue

            preds = pd.read_csv(pred_file, index_col=0)
            obs = pd.read_csv(obs_file, index_col=0)

            r2_fold = []
            for col in preds.columns:
                if col in obs.columns:
                    mask = obs[col].notna() & preds[col].notna()
                    if mask.sum() < 2:
                        r2_fold.append(np.nan)
                        continue
                    y = obs.loc[mask, col].values
                    yhat = preds.loc[mask, col].values
                    ss_res = np.sum((y - yhat) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2_fold.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
                else:
                    r2_fold.append(np.nan)
            all_r2.append(r2_fold)

        if all_r2:
            results[exp_type] = {
                'R2_mean': np.nanmean(all_r2, axis=0),
                'R2_std': np.nanstd(all_r2, axis=0),
            }

    return results


def load_2d_results(results_dir: Path):
    """Load all 2D experiment results."""
    results = {}
    for exp_dir in sorted(results_dir.iterdir()):
        summary_file = exp_dir / 'summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file, index_col=0)
            results[exp_dir.name] = df
    return results


def plot_comparison(results_2d: dict, baselines: dict, output_path: Path):
    """Generate comparison bar plot of R² across methods."""
    fig, ax = plt.subplots(figsize=(16, 8))

    methods = {}

    # Add baselines
    if 'multi' in baselines:
        methods['Cherif 1D-CNN'] = baselines['multi']['R2_mean']
    if 'plsr_sing' in baselines:
        methods['PLSR'] = baselines['plsr_sing']['R2_mean']

    # Add 2D results
    for name, df in results_2d.items():
        if 'R2_mean' in df.columns:
            methods[name] = df['R2_mean'].values

    if not methods:
        print("No results to plot.")
        return

    # Find common length
    min_len = min(len(v) for v in methods.values())
    traits = TRAIT_SHORT[:min_len]

    x = np.arange(min_len)
    width = 0.8 / len(methods)

    for i, (name, r2) in enumerate(methods.items()):
        ax.bar(x + i * width, r2[:min_len], width, label=name, alpha=0.85)

    ax.set_xlabel('Trait')
    ax.set_ylabel('R²')
    ax.set_title('Multi-trait R² Comparison: 2D Methods vs Baselines')
    ax.set_xticks(x + width * len(methods) / 2)
    ax.set_xticklabels(traits, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")


def generate_summary_table(results_2d: dict, baselines: dict, output_path: Path):
    """Generate CSV summary table of all methods."""
    rows = []

    if 'multi' in baselines:
        rows.append({
            'Method': 'Cherif 1D-CNN (multi)',
            'Mean R²': np.nanmean(baselines['multi']['R2_mean']),
            'Std R²': np.nanmean(baselines['multi']['R2_std']),
        })
    if 'plsr_sing' in baselines:
        rows.append({
            'Method': 'PLSR (single)',
            'Mean R²': np.nanmean(baselines['plsr_sing']['R2_mean']),
            'Std R²': np.nanmean(baselines['plsr_sing']['R2_std']),
        })

    for name, df in results_2d.items():
        if 'R2_mean' in df.columns:
            rows.append({
                'Method': name,
                'Mean R²': np.nanmean(df['R2_mean'].values),
                'Std R²': np.nanmean(df['R2_std'].values) if 'R2_std' in df.columns else np.nan,
            })

    summary = pd.DataFrame(rows).sort_values('Mean R²', ascending=False)
    summary.to_csv(output_path, index=False)
    print(f"\nSummary table saved to {output_path}")
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description='Compare 2D methods vs baselines')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--cherif-dir', type=str,
                        default='multi-traitretrieval/models')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cherif_dir = Path(args.cherif_dir)

    baselines = load_cherif_baselines(cherif_dir)
    results_2d = load_2d_results(results_dir)

    output_dir = results_dir / 'comparison'
    output_dir.mkdir(exist_ok=True)

    plot_comparison(results_2d, baselines, output_dir / 'r2_comparison.png')
    generate_summary_table(results_2d, baselines, output_dir / 'summary.csv')


if __name__ == '__main__':
    main()
