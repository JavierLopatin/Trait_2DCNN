"""Generate heatmap of pairwise statistical differences between transforms.

Uses GHS results: 3 seeds × 8 traits = 24 observations per pair.
Bonferroni-corrected alpha = 0.05/15 = 0.003.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# Transform names in performance order
names = ['Reshape', 'CWT', 'Spectrogram', 'GAF', '2D-COS', 'MTF']
n = len(names)

# Delta R² (row - col) from pending_experiments_response.txt
delta = np.full((n, n), np.nan)
# p-values
pvals = np.full((n, n), np.nan)

# Data: (row_idx, col_idx, delta, p-value)
pairs = [
    (0, 1, +0.043, 0.00033),   # reshape vs cwt
    (0, 2, +0.048, 0.00304),   # reshape vs spectrogram
    (0, 3, +0.075, 0.00000),   # reshape vs gaf
    (0, 4, +0.128, 0.00001),   # reshape vs cos2d
    (0, 5, +0.265, 0.00000),   # reshape vs mtf
    (1, 2, +0.005, 0.75894),   # cwt vs spectrogram
    (1, 3, +0.031, 0.03028),   # cwt vs gaf
    (1, 4, +0.085, 0.00007),   # cwt vs cos2d
    (1, 5, +0.221, 0.00000),   # cwt vs mtf
    (2, 3, +0.027, 0.13727),   # spectrogram vs gaf
    (2, 4, +0.081, 0.00494),   # spectrogram vs cos2d
    (2, 5, +0.217, 0.00000),   # spectrogram vs mtf
    (3, 4, -0.054, 0.02507),   # gaf vs cos2d
    (3, 5, +0.190, 0.00000),   # gaf vs mtf
    (4, 5, +0.136, 0.00000),   # cos2d vs mtf
]

for i, j, d, p in pairs:
    delta[i, j] = d
    delta[j, i] = -d
    pvals[i, j] = p
    pvals[j, i] = p

# Fill diagonal
np.fill_diagonal(delta, 0.0)
np.fill_diagonal(pvals, 1.0)

# Bonferroni threshold
alpha_bonf = 0.05 / 15  # 0.00333

# Create figure
fig, ax = plt.subplots(figsize=(7, 6))

# Diverging colormap centered at 0
vmax = 0.28
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(delta, cmap=cmap, norm=norm, aspect='equal')

# Add text annotations
for i in range(n):
    for j in range(n):
        if i == j:
            ax.text(j, i, '—', ha='center', va='center', fontsize=11,
                    color='gray')
        else:
            val = delta[i, j]
            sig = pvals[i, j] < alpha_bonf
            # Color text for readability
            text_color = 'white' if abs(val) > 0.15 else 'black'
            sign = '+' if val > 0 else ''
            label = f'{sign}{val:.3f}'
            if sig:
                label += '\n*'
            ax.text(j, i, label, ha='center', va='center', fontsize=10,
                    fontweight='bold' if sig else 'normal', color=text_color)

# Axes
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_yticklabels(names)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='$\Delta R^2$ (row $-$ column)')

# Caption note
ax.set_title('Pairwise $\Delta R^2$ between transforms\n(* = significant after Bonferroni correction, $\\alpha$ = 0.003)',
             fontsize=12, pad=12)

plt.tight_layout()
plt.savefig(OUT_DIR / 'pairwise_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'pairwise_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved pairwise_heatmap.pdf and .png')
