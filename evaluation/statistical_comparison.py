"""Statistical comparison of 2D transforms vs Cherif 1D-CNN baseline.

Usage:
    python -m evaluation.statistical_comparison                  # Cherif 2023 analysis
    python -m evaluation.statistical_comparison --pairwise       # + pairwise between transforms
    python -m evaluation.statistical_comparison --ghs-pairwise   # GHS pairwise (3 seeds)
"""
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from itertools import combinations
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_cherif_per_fold(data_dir='multi-traitretrieval/models'):
    """Load Cherif's per-fold R² from Predictions/Observations.

    Predictions columns have ' Predictions' suffix, observations don't.
    Match columns by trait name prefix (before the unit parenthetical).
    """
    # Map trait name prefixes to short names
    prefix_to_short = {
        'LMA': 'LMA', 'N content': 'N', 'LAI': 'LAI', 'C content': 'C',
        'Chl content': 'Chl', 'EWT': 'EWT', 'Carotenoid': 'Car',
        'Phosphorus': 'P', 'Lignin': 'Lignin', 'Cellulose': 'Cel',
        'Fiber': 'Fiber', 'Anthocyanin': 'Anth', 'NSC': 'NSC',
        'Magnesium': 'Mg', 'Ca content': 'Ca', 'Potassium': 'K',
        'Boron': 'B', 'Copper': 'Cu', 'Sulfur': 'S', 'Manganese': 'Mn',
    }

    def match_col(columns, prefix):
        """Find column matching a trait prefix."""
        for col in columns:
            if col.startswith(prefix):
                return col
        return None

    cherif_r2 = {}
    for fold in range(1, 6):
        preds = pd.read_csv(f'{data_dir}/multi_{fold}/inexact/Predictions.csv')
        obs = pd.read_csv(f'{data_dir}/multi_{fold}/inexact/Observations.csv')
        fold_r2 = {}
        for prefix, short in prefix_to_short.items():
            pred_col = match_col(preds.columns, prefix)
            obs_col = match_col(obs.columns, prefix)
            if pred_col and obs_col:
                mask = ~(preds[pred_col].isna() | obs[obs_col].isna())
                if mask.sum() > 2:
                    fold_r2[short] = r2_score(obs.loc[mask, obs_col],
                                              preds.loc[mask, pred_col])
        cherif_r2[fold] = fold_r2
    return cherif_r2


def load_2d_per_fold(results_dir='results'):
    """Load our 2D model per-fold R²."""
    transforms = ['reshape', 'cwt', 'gaf', 'cos2d', 'spectrogram']
    our_r2 = {}
    for t in transforms:
        our_r2[t] = {}
        for fold in range(1, 6):
            p = Path(results_dir) / f'{t}_efficientnet_b0' / f'fold_{fold}' / 'metrics.csv'
            if p.exists():
                df = pd.read_csv(p, index_col=0)
                our_r2[t][fold] = df['R2'].to_dict()
    return our_r2


def load_ghs_per_seed(results_dir='results'):
    """Load GHS per-seed R² for all transforms.

    Returns dict: {transform: {seed: {trait: R²}}}
    """
    transforms = ['reshape', 'cwt', 'spectrogram', 'cos2d', 'gaf', 'mtf']
    seeds = [155, 240, 318]
    ghs_r2 = {}
    for t in transforms:
        ghs_r2[t] = {}
        for seed in seeds:
            # First seed (155) has no suffix in directory name
            if seed == 155:
                p = Path(results_dir) / f'greenhs_{t}_efficientnet_b0' / 'fold_1' / 'metrics.csv'
            else:
                p = Path(results_dir) / f'greenhs_{t}_efficientnet_b0_s{seed}' / 'fold_1' / 'metrics.csv'
            if p.exists():
                df = pd.read_csv(p, index_col=0)
                ghs_r2[t][seed] = df['R2'].to_dict()
    return ghs_r2


def pairwise_transform_tests(r2_matrix, transforms, traits, folds_or_seeds,
                              dataset_label, alpha=0.05):
    """Pairwise paired t-tests between all transform pairs.

    Args:
        r2_matrix: {trait: {method: [R² per fold/seed]}}
        transforms: list of transform names
        traits: list of trait names
        folds_or_seeds: list of fold/seed indices
        dataset_label: string for display
        alpha: significance level
    """
    n_pairs = len(list(combinations(transforms, 2)))
    bonferroni_alpha = alpha / n_pairs

    print(f"\n{'=' * 90}")
    print(f"PAIRWISE PAIRED T-TESTS BETWEEN TRANSFORMS — {dataset_label}")
    print(f"{'=' * 90}")
    print(f"N pairs = {n_pairs}, Bonferroni α = {bonferroni_alpha:.4f}")
    print(f"Observations per pair: {len(traits)} traits × {len(folds_or_seeds)} folds/seeds "
          f"= {len(traits) * len(folds_or_seeds)} paired values\n")

    # Summary table header
    print(f"{'Transform A':>14s} vs {'Transform B':<14s} | {'Mean A':>7s} | {'Mean B':>7s} | "
          f"{'Delta':>7s} | {'t':>7s} | {'p':>9s} | {'p<α':>5s} | {'p<Bonf':>6s}")
    print("-" * 100)

    results = []
    for t_a, t_b in combinations(transforms, 2):
        vals_a, vals_b = [], []
        for trait in traits:
            if trait in r2_matrix and t_a in r2_matrix[trait] and t_b in r2_matrix[trait]:
                for i in range(len(folds_or_seeds)):
                    if i < len(r2_matrix[trait][t_a]) and i < len(r2_matrix[trait][t_b]):
                        va = r2_matrix[trait][t_a][i]
                        vb = r2_matrix[trait][t_b][i]
                        if not (np.isnan(va) or np.isnan(vb)):
                            vals_a.append(va)
                            vals_b.append(vb)

        if len(vals_a) < 3:
            continue

        a, b = np.array(vals_a), np.array(vals_b)
        mean_a, mean_b = np.mean(a), np.mean(b)
        delta = mean_a - mean_b
        t_stat, p_val = stats.ttest_rel(a, b)
        sig_raw = "Yes" if p_val < alpha else "No"
        sig_bonf = "Yes" if p_val < bonferroni_alpha else "No"

        print(f"{t_a:>14s} vs {t_b:<14s} | {mean_a:>7.3f} | {mean_b:>7.3f} | "
              f"{delta:>+7.3f} | {t_stat:>7.3f} | {p_val:>9.5f} | {sig_raw:>5s} | {sig_bonf:>6s}")

        results.append({
            'A': t_a, 'B': t_b, 'mean_A': mean_a, 'mean_B': mean_b,
            'delta': delta, 't': t_stat, 'p': p_val,
            'sig': p_val < alpha, 'sig_bonf': p_val < bonferroni_alpha,
            'n': len(vals_a)
        })

    # Overall ranking summary
    print(f"\n--- Ranking summary (by mean R² across paired observations) ---")
    means = {}
    for t in transforms:
        vals = []
        for trait in traits:
            if trait in r2_matrix and t in r2_matrix[trait]:
                vals.extend([v for v in r2_matrix[trait][t] if not np.isnan(v)])
        if vals:
            means[t] = np.mean(vals)
    ranked = sorted(means.items(), key=lambda x: -x[1])
    for rank, (t, m) in enumerate(ranked, 1):
        print(f"  {rank}. {t:<14s} R² = {m:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairwise', action='store_true',
                        help='Run pairwise tests between transforms (Cherif 2023)')
    parser.add_argument('--ghs-pairwise', action='store_true',
                        help='Run pairwise tests on GHS (3 seeds)')
    parser.add_argument('--all', action='store_true',
                        help='Run all analyses')
    args = parser.parse_args()
    cherif_r2 = load_cherif_per_fold()
    our_r2 = load_2d_per_fold()

    transforms = [t for t in ['reshape', 'cwt', 'gaf', 'cos2d', 'spectrogram'] if t in our_r2]
    traits = ['LMA', 'LAI', 'N', 'C', 'Chl', 'Car', 'Anth', 'EWT', 'Cel', 'Lignin',
              'Fiber', 'NSC', 'P', 'Ca', 'Mg', 'K', 'S', 'B', 'Cu', 'Mn']
    all_methods = ['cherif_1dcnn'] + transforms

    # Build R² matrix
    r2_matrix = {}
    for trait in traits:
        r2_matrix[trait] = {}
        r2_matrix[trait]['cherif_1dcnn'] = [cherif_r2[f].get(trait, np.nan) for f in range(1, 6)]
        for t in transforms:
            r2_matrix[trait][t] = [our_r2[t][f].get(trait, np.nan) for f in range(1, 6)]

    # ========== PART 1: Mean R² table ==========
    print("=" * 85)
    print("MEAN R² ACROSS 5 FOLDS (* = best)")
    print("=" * 85)
    labels = {'cherif_1dcnn': 'Cherif', 'reshape': 'Reshp', 'cwt': 'CWT',
              'gaf': 'GAF', 'cos2d': 'COS2D', 'spectrogram': 'Spctro'}
    header = f"{'Trait':>8s}"
    for m in all_methods:
        header += f" | {labels[m]:>7s}"
    print(header)
    print("-" * 85)
    for trait in traits:
        vals = [np.nanmean(r2_matrix[trait][m]) for m in all_methods]
        best_idx = np.nanargmax(vals)
        parts = [f"{trait:>8s}"]
        for i, v in enumerate(vals):
            marker = "*" if i == best_idx else " "
            parts.append(f"{v:>6.3f}{marker}")
        print(" | ".join(parts))
    print("-" * 85)
    overall = [np.nanmean([np.nanmean(r2_matrix[t][m]) for t in traits]) for m in all_methods]
    parts = [f"{'MEAN':>8s}"]
    for v in overall:
        parts.append(f"{v:>7.4f}")
    print(" | ".join(parts))

    # ========== Paired t-tests per trait ==========
    print("\n" + "=" * 85)
    print("PAIRED T-TESTS vs CHERIF (per trait, α=0.05)")
    print("+ = sig. better, - = sig. worse")
    print("=" * 85)

    t_labels = {'reshape': 'Reshape', 'cwt': 'CWT', 'gaf': 'GAF',
                'cos2d': 'COS2D', 'spectrogram': 'Spectrogram'}
    for t_name in transforms:
        t_label = t_labels[t_name]
        print(f"\n--- {t_label} vs Cherif ---")
        print(f"{'Trait':>8s} | {'Cherif':>7s} | {t_label[:7]:>7s} | {'Delta':>7s} | {'t':>6s} | {'p':>7s} | Sig")
        print("-" * 65)
        sb, sw = 0, 0
        for trait in traits:
            c = np.array(r2_matrix[trait]['cherif_1dcnn'])
            o = np.array(r2_matrix[trait][t_name])
            dm = np.mean(o - c)
            ts, pv = stats.ttest_rel(o, c)
            sig = "ns"
            if pv < 0.05:
                sig = " +" if dm > 0 else " -"
                if dm > 0:
                    sb += 1
                else:
                    sw += 1
            print(f"{trait:>8s} | {np.mean(c):>7.3f} | {np.mean(o):>7.3f} | {dm:>+7.3f} | {ts:>6.2f} | {pv:>7.4f} | {sig}")
        print(f"  Sig. better: {sb}, Sig. worse: {sw}, NS: {20 - sb - sw}")

    # Overall test
    print("\n" + "=" * 85)
    print("OVERALL PAIRED T-TEST (mean R² across 20 traits per fold)")
    print("=" * 85)
    for t_name in transforms:
        t_label = t_labels[t_name]
        cf, of = [], []
        for fold in range(5):
            cf.append(np.nanmean([r2_matrix[t]['cherif_1dcnn'][fold] for t in traits]))
            of.append(np.nanmean([r2_matrix[t][t_name][fold] for t in traits]))
        c, o = np.array(cf), np.array(of)
        ts, pv = stats.ttest_rel(o, c)
        sig = " *" if pv < 0.05 else " ns"
        print(f"{t_label:>12s}: Cherif={np.mean(c):.4f}, Ours={np.mean(o):.4f}, "
              f"D={np.mean(o - c):+.4f}, t={ts:.3f}, p={pv:.4f}{sig}")

    # ========== PART 2: Multi-channel ==========
    print("\n\n" + "=" * 85)
    print("MULTI-CHANNEL COMBINATION ANALYSIS")
    print("=" * 85)

    n_ch = {'reshape': 1, 'cwt': 1, 'cos2d': 2, 'gaf': 2, 'spectrogram': 3}

    # Correlation matrix
    all_flat = {}
    for t in transforms:
        vals = []
        for trait in traits:
            vals.extend(r2_matrix[trait][t])
        all_flat[t] = vals
    corr_df = pd.DataFrame(all_flat).corr()

    print("\nCorrelation between transforms (trait-fold R²):")
    print("Lower correlation = more complementary")
    print(corr_df.round(3).to_string())

    # Pair scores
    print("\n\nPair combinations ranked by complementarity:")
    print(f"{'Combo':>28s} | {'Corr':>5s} | {'OracleR²':>8s} | {'Score':>6s} | {'Ch':>3s}")
    print("-" * 60)
    combo2 = []
    for t1, t2 in combinations(transforms, 2):
        corr = corr_df.loc[t1, t2]
        oracle = np.mean([max(r2_matrix[trait][t1][f], r2_matrix[trait][t2][f])
                          for trait in traits for f in range(5)])
        score = oracle * (1 - corr)
        combo2.append((t1, t2, corr, oracle, score, n_ch[t1] + n_ch[t2]))
    combo2.sort(key=lambda x: -x[4])
    for t1, t2, c, o, s, ch in combo2:
        print(f"{t1 + ' + ' + t2:>28s} | {c:>5.3f} | {o:>8.3f} | {s:>6.3f} | {ch:>3d}")

    # Triple combos
    print("\nTriple combinations (top 5):")
    print(f"{'Combo':>35s} | {'MCorr':>5s} | {'OracleR²':>8s} | {'Score':>6s} | {'Ch':>3s}")
    print("-" * 65)
    combo3 = []
    for t1, t2, t3 in combinations(transforms, 3):
        mc = np.mean([corr_df.loc[t1, t2], corr_df.loc[t1, t3], corr_df.loc[t2, t3]])
        oracle = np.mean([max(r2_matrix[trait][t1][f], r2_matrix[trait][t2][f],
                              r2_matrix[trait][t3][f])
                          for trait in traits for f in range(5)])
        score = oracle * (1 - mc)
        combo3.append((f"{t1}+{t2}+{t3}", mc, oracle, score,
                        n_ch[t1] + n_ch[t2] + n_ch[t3]))
    combo3.sort(key=lambda x: -x[3])
    for name, mc, o, s, ch in combo3[:5]:
        print(f"{name:>35s} | {mc:>5.3f} | {o:>8.3f} | {s:>6.3f} | {ch:>3d}")

    # Best method per trait per fold
    print("\n\n" + "=" * 85)
    print("BEST METHOD PER TRAIT PER FOLD")
    print("=" * 85)
    print(f"{'Trait':>8s} | {'F1':>8s} | {'F2':>8s} | {'F3':>8s} | {'F4':>8s} | {'F5':>8s} | Consensus")
    print("-" * 85)
    for trait in traits:
        best_per_fold = []
        for fold in range(5):
            vals = {m: r2_matrix[trait][m][fold] for m in transforms}
            best_per_fold.append(max(vals, key=vals.get))
        counts = Counter(best_per_fold)
        consensus = counts.most_common(1)[0]
        parts = [f"{trait:>8s}"]
        for b in best_per_fold:
            parts.append(f"{b:>8s}")
        parts.append(f"{consensus[0]} ({consensus[1]}/5)")
        print(" | ".join(parts))

    # ========== PAIRWISE TRANSFORM TESTS — Cherif 2023 ==========
    if args.pairwise or args.all:
        pairwise_transform_tests(
            r2_matrix, transforms, traits,
            folds_or_seeds=list(range(5)),
            dataset_label="Cherif 2023 (5-fold CV, 20 traits)"
        )

    # ========== PAIRWISE TRANSFORM TESTS — GHS ==========
    if args.ghs_pairwise or args.all:
        ghs_r2 = load_ghs_per_seed()
        ghs_traits = ['Cab', 'Car', 'Anth', 'Cw', 'Cm', 'LAI', 'Cp', 'Cbc']
        ghs_transforms = [t for t in ['reshape', 'cwt', 'spectrogram', 'cos2d', 'gaf', 'mtf']
                          if len(ghs_r2.get(t, {})) >= 2]
        seeds = [155, 240, 318]

        # Build GHS R² matrix: {trait: {transform: [R² per seed]}}
        ghs_matrix = {}
        for trait in ghs_traits:
            ghs_matrix[trait] = {}
            for t in ghs_transforms:
                ghs_matrix[trait][t] = []
                for seed in seeds:
                    if seed in ghs_r2[t]:
                        ghs_matrix[trait][t].append(ghs_r2[t][seed].get(trait, np.nan))

        # Print available transforms
        print(f"\n\nGHS transforms with ≥2 seeds: {ghs_transforms}")
        for t in ghs_transforms:
            n_seeds = len(ghs_r2[t])
            mean_r2 = np.nanmean([ghs_r2[t][s].get(trait, np.nan)
                                   for s in ghs_r2[t] for trait in ghs_traits])
            print(f"  {t}: {n_seeds} seeds, mean R² = {mean_r2:.3f}")

        pairwise_transform_tests(
            ghs_matrix, ghs_transforms, ghs_traits,
            folds_or_seeds=seeds,
            dataset_label="GreenHyperSpectra (3 seeds, 8 traits)"
        )


if __name__ == '__main__':
    main()
