"""Generate Figure: example spectra and their 2D transformations.

Panel A: 3 spectra with different trait combinations.
Panel B: 6 transforms × 3 spectra grid.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))

from transforms import (
    ReshapeTransform, CWTTransform, SpectrogramTransform,
    COS2DTransform, GAFTransform, MTFTransform,
)

OUT_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE / 'data/GreenHyperSpectra/labeled_test.csv'
SIZE = 224

# Global font settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 14,
})

# GHS traits and spectral columns
TRAIT_COLS = ['cab', 'car', 'anth', 'cw', 'cm', 'LAI', 'cp', 'cbc']

# ── Load data ──
df = pd.read_csv(TEST_FILE)
spec_cols = [c for c in df.columns if c not in TRAIT_COLS + ['dataset']]
wavelengths = np.array([float(c) for c in spec_cols])

# ── Select 3 diverse spectra ──
# Pick samples with contrasting trait profiles:
# 1) High Cab + high LAI (dense green canopy)
# 2) Low Cab + high Cm (senescent/structural)
# 3) High Anth + moderate LAI (stressed/colored)
traits = df[TRAIT_COLS]

# Normalize traits for ranking
tn = (traits - traits.min()) / (traits.max() - traits.min())

idx1 = tn.dropna(subset=['cab', 'LAI']).eval('cab + LAI').idxmax()
idx2 = tn.dropna(subset=['cab', 'cm']).eval('-cab + cm').idxmax()
idx3 = tn.dropna(subset=['anth', 'LAI']).eval('anth + 0.5*LAI').idxmax()

indices = [idx1, idx2, idx3]
labels = ['High Chl + high LAI', 'Low Chl + high LMA', 'High Anth']
colors = ['#0072B2', '#D55E00', '#009E73']  # colorblind-friendly (blue, vermillion, green)

spectra = df.loc[indices, spec_cols].values.astype(np.float32)

# ── Transforms ──
transforms = [
    ('Reshape', ReshapeTransform(output_size=SIZE)),
    ('CWT', CWTTransform(output_size=SIZE)),
    ('Spectrogram', SpectrogramTransform(output_size=SIZE)),
    ('2D-COS', COS2DTransform(output_size=SIZE)),
    ('GAF', GAFTransform(output_size=SIZE)),
    ('MTF', MTFTransform(output_size=SIZE)),
]

# ── Figure ──
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 6, height_ratios=[1.2, 1, 1],
                       hspace=0.3, wspace=0.25)

# ── Panel A: spectra (top row, spanning all 6 columns) ──
ax_spec = fig.add_subplot(gs[0, :])
for j, (idx, label, color) in enumerate(zip(indices, labels, colors)):
    ax_spec.plot(wavelengths, spectra[j], color=color, lw=1.2,
                 label=f'Spectrum {j+1}: {label}', alpha=0.85)
ax_spec.set_xlabel('Wavelength (nm)', fontsize=13)
ax_spec.set_ylabel('Reflectance', fontsize=13)
ax_spec.legend(fontsize=12, loc='upper right')
ax_spec.tick_params(labelsize=11)
ax_spec.set_xlim(400, 2450)
ax_spec.text(-0.04, 1.05, '(a)', transform=ax_spec.transAxes,
             fontsize=15, fontweight='bold', va='bottom')

# ── Panel B: transforms grid (rows 1-2 = spectra 1-3 interleaved) ──
# Layout: 2 rows × 6 cols. Row 0 = spectrum 1 images, Row 1 = spectrum 2 images
# But we have 3 spectra × 6 transforms = 18 images → not enough space.
# Better: 6 cols (transforms) × 3 rows (spectra) below panel A.
# Rearrange: use a separate gridspec for panel B.

fig.clear()
gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.5, 2.5], hspace=0.25)

# Panel A
ax_spec = fig.add_subplot(gs_main[0])
for j, (idx, label, color) in enumerate(zip(indices, labels, colors)):
    ax_spec.plot(wavelengths, spectra[j], color=color, lw=2.2,
                 label=f'Spectrum {j+1}: {label}', alpha=0.9)
ax_spec.set_xlabel('Wavelength (nm)')
ax_spec.set_ylabel('Reflectance')
ax_spec.legend(loc='upper right')
ax_spec.set_xlim(400, 2450)
ax_spec.text(-0.04, 1.05, '(a)', transform=ax_spec.transAxes,
             fontsize=17, fontweight='bold', va='bottom')

# Panel B: 3 rows (spectra) × 6 cols (transforms)
gs_b = gridspec.GridSpecFromSubplotSpec(3, 6, subplot_spec=gs_main[1],
                                        hspace=0.3, wspace=0.15)

for row in range(3):
    spectrum = spectra[row]
    for col, (tname, tfm) in enumerate(transforms):
        ax = fig.add_subplot(gs_b[row, col])
        img = tfm.transform(spectrum)

        if img.ndim == 3:
            if img.shape[-1] == 3:
                # Spectrogram: log-scale then normalize per channel
                img_disp = np.log1p(img * 1000).copy()
                for ch in range(3):
                    ch_min, ch_max = img_disp[:,:,ch].min(), img_disp[:,:,ch].max()
                    if ch_max > ch_min:
                        img_disp[:,:,ch] = (img_disp[:,:,ch] - ch_min) / (ch_max - ch_min)
                ax.imshow(img_disp, aspect='equal')
            elif img.shape[-1] == 2:
                # 2-channel (2D-COS, GAF): map ch1→red, ch2→blue
                ch1 = img[:, :, 0]
                ch2 = img[:, :, 1]
                rgb = np.zeros((*ch1.shape, 3), dtype=np.float32)
                rgb[:, :, 0] = ch1  # red
                rgb[:, :, 1] = 0.3 * (ch1 + ch2)  # green tint
                rgb[:, :, 2] = ch2  # blue
                # Normalize to [0, 1]
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                ax.imshow(rgb, aspect='equal')
            else:
                ax.imshow(img[:, :, 0], cmap='viridis', aspect='equal')
        else:
            ax.imshow(img, cmap='viridis', aspect='equal')

        ax.set_xticks([])
        ax.set_yticks([])

        # Column title (transform name) on top row only
        if row == 0:
            ax.set_title(tname, fontsize=14, fontweight='bold')

        # Row label (spectrum number) on left column only
        if col == 0:
            ax.set_ylabel(f'Sp. {row+1}', fontsize=14, color=colors[row],
                          fontweight='bold')

# Panel B label — align with (a) using same axes-relative x position
fig.text(ax_spec.get_position().x0 - 0.04, 0.60, '(b)', fontsize=17,
         fontweight='bold', va='bottom')

fig.set_size_inches(18, 14)
plt.savefig(OUT_DIR / 'transforms_example.pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'transforms_example.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved transforms_example.pdf and .png')
