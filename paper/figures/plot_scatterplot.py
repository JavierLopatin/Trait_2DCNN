"""Generate scatterplot of observed vs predicted for the best model (Reshape + EfficientNet-B0)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent.parent.parent
PRED_FILE = BASE / 'results/greenhs_reshape_efficientnet_b0/fold_1/predictions.csv'
TEST_FILE = BASE / 'data/GreenHyperSpectra/labeled_test.csv'
METRICS_FILE = BASE / 'results/greenhs_reshape_efficientnet_b0/fold_1/metrics.csv'
OUT_DIR = Path(__file__).resolve().parent

# Trait mapping: predictions columns -> test CSV columns
TRAIT_MAP = {
    'Cab': 'cab', 'Car': 'car', 'Anth': 'anth', 'Cw': 'cw',
    'Cm': 'cm', 'LAI': 'LAI', 'Cp': 'cp', 'Cbc': 'cbc',
}
TRAIT_LABELS = {
    'Cab': r'C$_{ab}$ ($\mu$g cm$^{-2}$)',
    'Car': r'C$_{ar}$ ($\mu$g cm$^{-2}$)',
    'Anth': r'C$_{anth}$ ($\mu$g cm$^{-2}$)',
    'Cw': r'C$_w$ (g m$^{-2}$)',
    'Cm': r'C$_m$ (g m$^{-2}$)',
    'LAI': r'LAI (m$^2$ m$^{-2}$)',
    'Cp': r'C$_p$ (g m$^{-2}$)',
    'Cbc': r'C$_{bc}$ (g m$^{-2}$)',
}
TRAIT_TITLES = {
    'Cab': 'Chlorophyll a+b',
    'Car': 'Carotenoids',
    'Anth': 'Anthocyanins',
    'Cw': 'Equivalent water thickness',
    'Cm': 'Leaf mass per area',
    'LAI': 'Leaf area index',
    'Cp': 'Protein content',
    'Cbc': 'Carbon-based constituents',
}

# Load data
preds = pd.read_csv(PRED_FILE)
test_df = pd.read_csv(TEST_FILE)
metrics = pd.read_csv(METRICS_FILE, index_col=0)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, (pred_col, obs_col) in enumerate(TRAIT_MAP.items()):
    ax = axes[i]
    y_pred = preds[pred_col].values
    y_obs = test_df[obs_col].values

    # Mask NaN observations
    mask = np.isfinite(y_obs) & np.isfinite(y_pred)
    y_obs_m, y_pred_m = y_obs[mask], y_pred[mask]

    # KDE coloring
    xy = np.vstack([y_obs_m, y_pred_m])
    kde = gaussian_kde(xy)(xy)

    ax.scatter(y_obs_m, y_pred_m, c=kde, s=18, alpha=0.7, cmap='viridis',
               edgecolors='none', rasterized=True)

    # 1:1 line
    lo = min(y_obs_m.min(), y_pred_m.min())
    hi = max(y_obs_m.max(), y_pred_m.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            'k--', lw=1, alpha=0.6)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    # Metrics annotation
    r2 = metrics.loc[pred_col, 'R2']
    rmse = metrics.loc[pred_col, 'RMSE']
    obs_range = y_obs_m.max() - y_obs_m.min()
    nrmse = (rmse / obs_range) * 100 if obs_range > 0 else 0
    ax.text(0.05, 0.92, f'$R^2$ = {r2:.3f}\nnRMSE = {nrmse:.1f}%',
            transform=ax.transAxes, fontsize=11, verticalalignment='top')

    ax.set_title(TRAIT_TITLES[pred_col], fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Observed {TRAIT_LABELS[pred_col]}', fontsize=12)
    ax.set_ylabel(f'Predicted {TRAIT_LABELS[pred_col]}', fontsize=12)
    ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(OUT_DIR / 'scatterplot_best_model.pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'scatterplot_best_model.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved scatterplot_best_model.pdf and .png')
