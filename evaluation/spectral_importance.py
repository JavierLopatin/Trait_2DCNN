"""Spectral variable importance for the Reshape 2D-CNN via attribution methods.

Because the Reshape transform preserves band order (pixel (i,j) <-> band
k = 42*i + j), attribution maps computed on the 2D image can be "unfolded" back
to the 1D wavelength axis. This links the CNN's relevance to the classical
band/variable selection of remote sensing: which spectral regions predict each
trait best.

Two complementary, standard CNN-interpretability paradigms are used:
  - Integrated Gradients (IG): input-level attribution on the (normalized) image
    the CNN actually receives, then unfolded to the 1721-band axis. Signed.
  - Grad-CAM: relevance over the final conv embedding (7x7x1280), region-level.

Why attribute on the normalized image rather than the raw spectrum: the Reshape
transform applies a per-image min-max scaling, and since zero-padding pins
vmin~=0 this is effectively a division by the reflectance maximum (the NIR
plateau ~1073 nm). Differentiating through that scaling makes the gradient pile
up on the few bands that set the scale, a trait-agnostic artifact. Attributing
on the network input (the normalized image) is the canonical IG setup, keeps the
completeness axiom, and maps exactly to bands via the inverse Reshape.

Usage:
    python -m evaluation.spectral_importance --method both --n-samples 200
    python -m evaluation.spectral_importance --method both     # full set, 3 seeds
"""
import argparse
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from training.config import GHS_DATA_DIR, GHS_TRAIT_NAMES, GHS_TRAIT_SHORT, N_BANDS
from training.data_loader import _split_trait_spectral_columns, load_ghs_data
from training.lightning_module import TraitRegressionModule
from models.trait_model import get_model


# --- Wavelength axis ------------------------------------------------------

@functools.lru_cache(maxsize=1)
def wavelengths() -> np.ndarray:
    """Real wavelength vector (nm) from the GHS spectral columns.

    1721 bands over 400-2450 nm with two water-band gaps removed
    (1351-1430 and 1801-2050 nm), so the axis is NOT contiguous.
    """
    df = pd.read_csv(GHS_DATA_DIR / 'labeled_test.csv', nrows=0)
    _, spec_cols = _split_trait_spectral_columns(df, GHS_TRAIT_NAMES)
    return np.array([float(c) for c in spec_cols])


# Water-band gaps removed during preprocessing (for shading in plots).
WATER_GAPS = [(1350, 1431), (1800, 2051)]

# Known absorption regions per trait (lo_nm, hi_nm, label). Approximate ranges
# from Curran (1989), Kokaly et al. (2009), Elvidge (1990), Feret et al. (2008).
ABSORPTION_REGIONS = {
    'Cab':  [(400, 450, 'Chl Soret'), (640, 690, 'Chl b/a'), (690, 750, 'red-edge')],
    'Car':  [(440, 520, 'carotenoid')],
    'Anth': [(500, 560, 'anthocyanin')],
    'Cw':   [(950, 1000, 'water 970'), (1150, 1260, 'water 1200')],
    'Cm':   [(1700, 1780, 'dry matter'), (2050, 2140, 'cellulose'), (2260, 2350, 'lignin')],
    'LAI':  [(700, 1300, 'NIR canopy')],
    'Cp':   [(1480, 1530, 'protein N-H'), (2040, 2070, 'protein'),
             (2160, 2200, 'protein'), (2290, 2350, 'protein')],
    'Cbc':  [(1680, 1740, 'lignin'), (2080, 2140, 'cellulose'), (2260, 2350, 'cell./lignin')],
}


# --- Reshape pipeline (image generation) ----------------------------------

class DifferentiableReshape(nn.Module):
    """Torch reimplementation of ReshapeTransform.

    pad(n_bands -> side^2) -> reshape(side, side) -> bilinear interpolate
    (align_corners=True == scipy zoom order=1, validated identical) ->
    per-image min-max normalize. Used here to generate the normalized images
    the CNN receives (under no_grad); attribution is then taken w.r.t. those
    images, so the min-max step is not in the attribution graph.
    """

    def __init__(self, n_bands: int = N_BANDS, output_size: int = 224):
        super().__init__()
        self.n_bands = n_bands
        self.side = int(np.ceil(np.sqrt(n_bands)))
        self.output_size = output_size
        self.pad_n = self.side * self.side - n_bands

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = F.pad(spec, (0, self.pad_n))                  # (N, side^2)
        x = x.reshape(-1, 1, self.side, self.side)        # (N, 1, side, side)
        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode='bilinear', align_corners=True)
        vmin = x.amin(dim=(2, 3), keepdim=True)
        vmax = x.amax(dim=(2, 3), keepdim=True)
        return (x - vmin) / (vmax - vmin).clamp_min(1e-12)


def make_images(spectra: np.ndarray, device: str) -> torch.Tensor:
    """Generate normalized Reshape images (N,1,224,224) from raw spectra."""
    reshape = DifferentiableReshape().to(device)
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(spectra, dtype=np.float32)).to(device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return reshape(x).cpu()


# --- Model loading --------------------------------------------------------

def load_model(exp_dir: Path, device: str) -> nn.Module:
    """Load the trained EfficientNet-B0 backbone from a GHS reshape checkpoint."""
    ckpt = exp_dir / 'fold_1' / 'best.ckpt'
    model = get_model('efficientnet_b0', in_channels=1, n_traits=len(GHS_TRAIT_SHORT))
    lit = TraitRegressionModule.load_from_checkpoint(
        ckpt, model=model, map_location=device)
    return lit.model.to(device).eval()


# --- Integrated Gradients (on the normalized image input) -----------------

def integrated_gradients(model, images, baseline, target, n_steps=50,
                         batch_size=16, device='cpu'):
    """IG attribution of trait `target` w.r.t. each input pixel.

    images: (N,1,H,W) normalized images; baseline: (1,1,H,W). Returns (N,1,H,W)
    signed attributions via a Riemann sum over n_steps.
    """
    model.eval()
    b = baseline.to(device)
    out = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device)
        grad_sum = torch.zeros_like(x)
        for s in range(1, n_steps + 1):
            xi = (b + (s / n_steps) * (x - b)).detach().requires_grad_(True)
            score = model(xi)[:, target].sum()
            grad_sum += torch.autograd.grad(score, xi)[0]
        out.append(((x - b) * grad_sum / n_steps).detach().cpu())
    return torch.cat(out)


def ig_completeness(model, images, baseline, target, attr, device='cpu', k=16):
    """Max error of the IG completeness axiom: sum(attr) ~= f(x) - f(baseline)."""
    model.eval()
    with torch.no_grad():
        fb = model(baseline.to(device))[0, target].item()
        fx = model(images[:k].to(device))[:, target].cpu().numpy()
    summed = attr[:k].sum(dim=(1, 2, 3)).numpy() + fb
    return float(np.abs(summed - fx).max())


# --- Grad-CAM (embedding-level relevance) ---------------------------------

def grad_cam(model, images, target, target_layer, batch_size=32, device='cpu'):
    """Grad-CAM over the final conv embedding. Returns (N, h, w) heatmaps."""
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.update(v=o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.update(v=go[0]))
    cams = []
    try:
        for i in range(0, len(images), batch_size):
            x = images[i:i + batch_size].to(device)
            model.zero_grad(set_to_none=True)
            model(x)[:, target].sum().backward()
            A, G = acts['v'], grads['v']                 # (B, C, h, w)
            w = G.mean(dim=(2, 3), keepdim=True)         # channel weights
            cam = F.relu((w * A).sum(dim=1))             # (B, h, w)
            cams.append(cam.detach().cpu())
    finally:
        h1.remove()
        h2.remove()
    return torch.cat(cams)


# --- Unfold a 2D map back to the 1D band axis -----------------------------

def unfold_2d_to_bands(attr2d, n_bands=N_BANDS, side=42):
    """Project a 2D relevance map (H, W) onto the n_bands axis.

    Resizes to side x side (inverse of the forward interpolation), flattens
    row-major, drops the zero-padding tail.
    """
    from scipy.ndimage import zoom
    if attr2d.shape[0] != side:
        attr2d = zoom(attr2d, side / attr2d.shape[0], order=1)
    return attr2d.reshape(-1)[:n_bands]


def attribution_to_bands(attr):
    """(N,1,H,W) signed attribution -> (importance, signed) per band (n_bands,).

    importance = mean over samples of |attr|, unfolded; signed = mean of attr.
    """
    a = attr.squeeze(1).numpy()
    return unfold_2d_to_bands(np.abs(a).mean(0)), unfold_2d_to_bands(a.mean(0))


# --- Plotting -------------------------------------------------------------

def _broken(wl, y):
    """Insert NaN at water-band gaps so lines break instead of bridging them."""
    gaps = np.where(np.diff(wl) > 2)[0]
    wl2, y2 = wl.astype(float).copy(), y.astype(float).copy()
    for g in reversed(gaps):
        wl2 = np.insert(wl2, g + 1, np.nan)
        y2 = np.insert(y2, g + 1, np.nan)
    return wl2, y2


def _shade_context(ax, trait):
    """Shade removed water bands (grey) and known absorption regions (green)."""
    for lo, hi in WATER_GAPS:
        ax.axvspan(lo, hi, color='0.85', alpha=0.6, zorder=0)
    for lo, hi, _ in ABSORPTION_REGIONS.get(trait, []):
        ax.axvspan(lo, hi, color='tab:green', alpha=0.15, zorder=0)


def _overlay_cam(ax, cam, base_img):
    """Show a 2D Grad-CAM heatmap over a grayscale base image."""
    cam = cam / (cam.max() + 1e-12)
    up = F.interpolate(torch.tensor(cam)[None, None].float(),
                       size=base_img.shape, mode='bilinear', align_corners=True)[0, 0].numpy()
    ax.imshow(base_img, cmap='gray')
    ax.imshow(up, cmap='jet', alpha=0.5)
    ax.axis('off')


def plot_ig_grid(wl, imp_mean, imp_std, out):
    """Grid of per-trait IG importance profiles vs wavelength."""
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    for ax, trait, m, s in zip(axes.ravel(), GHS_TRAIT_SHORT, imp_mean, imp_std):
        peak = m.max() + 1e-12
        y, sd = m / peak, s / peak
        _shade_context(ax, trait)
        wlb, yb = _broken(wl, y)
        _, sb = _broken(wl, sd)
        ax.fill_between(wlb, np.clip(yb - sb, 0, None), yb + sb, color='tab:blue', alpha=0.2)
        ax.plot(wlb, yb, color='tab:blue', lw=1.1)
        ax.set_title(trait, fontsize=11, loc='left')
        ax.set_ylim(0, 1.08)
        ax.set_xlim(400, 2450)
    for ax in axes[-1]:
        ax.set_xlabel('Wavelength (nm)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Norm. |IG|')
    handles = [mpatches.Patch(color='tab:green', alpha=0.3, label='Known absorption region'),
               mpatches.Patch(color='0.85', label='Removed water band')]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=9, frameon=False)
    fig.suptitle('Integrated Gradients band importance (mean +/- std over seeds)', y=0.965)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_gradcam_2d(cam_per_trait, mean_img, out):
    """Grid of Grad-CAM heatmaps over the mean Reshape image."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, trait, cam in zip(axes.ravel(), GHS_TRAIT_SHORT, cam_per_trait):
        _overlay_cam(ax, cam, mean_img)
        ax.set_title(trait)
    fig.suptitle('Grad-CAM over final conv embedding (overlaid on mean Reshape image)')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_linking(trait_idx, wl, ig_mean, cam2d, mean_img, out):
    """Link the 2D image (Grad-CAM, left) to the 1D band profile (IG, right).

    Grad-CAM is shown only as the 2D heatmap: it is a low-resolution (7x7)
    embedding map, valuable as a spatial view of where the CNN looks, but
    aliased into a comb pattern if forced into a fine 1D curve. The fine
    per-band profile is provided by IG.
    """
    trait = GHS_TRAIT_SHORT[trait_idx]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 4.2),
                                   gridspec_kw={'width_ratios': [1, 2.4]})
    _overlay_cam(axL, cam2d, mean_img)
    axL.set_title(f'{trait}: Grad-CAM on 2D image (embedding)')

    y = ig_mean / (ig_mean.max() + 1e-12)
    _shade_context(axR, trait)
    wlb, yb = _broken(wl, y)
    axR.plot(wlb, yb, color='tab:blue', lw=1.2)
    axR.set_xlabel('Wavelength (nm)')
    axR.set_ylabel('Norm. |IG|')
    axR.set_xlim(400, 2450)
    axR.set_title(f'{trait}: IG band importance (unfolded to spectrum)')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- Enrichment in known absorption regions -------------------------------

def compute_enrichment(df):
    """Per-trait share of IG importance inside known absorption regions and its
    enrichment over chance (= fraction of bands those regions cover). Returns a
    DataFrame with trait, imp_frac, width_frac, enrichment, peak_nm.
    """
    wl = df['wavelength_nm'].values
    rows = []
    for tr in GHS_TRAIT_SHORT:
        imp = df[f'{tr}_imp_mean'].values
        inreg = np.zeros(len(wl), dtype=bool)
        for lo, hi, _ in ABSORPTION_REGIONS.get(tr, []):
            inreg |= (wl >= lo) & (wl <= hi)
        imp_frac = imp[inreg].sum() / imp.sum()
        width_frac = float(inreg.mean())
        rows.append({'trait': tr, 'imp_frac': imp_frac, 'width_frac': width_frac,
                     'enrichment': imp_frac / width_frac if width_frac else np.nan,
                     'peak_nm': float(wl[imp.argmax()])})
    return pd.DataFrame(rows)


def plot_enrichment(df, out):
    """Lollipop chart of per-trait enrichment of IG importance in known regions."""
    from matplotlib.colors import TwoSlopeNorm
    e = compute_enrichment(df).sort_values('enrichment').reset_index(drop=True)
    vmax = max(2.0, float(e['enrichment'].max()))
    colors = plt.cm.RdYlGn(TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=vmax)(e['enrichment'].values))
    y = np.arange(len(e))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.axvline(1.0, color='0.45', ls='--', lw=1.2, zorder=1)
    ax.text(1.0, len(e) - 0.3, '  chance', color='0.45', fontsize=9, va='top', style='italic')
    for yi, (enr, c) in enumerate(zip(e['enrichment'], colors)):
        ax.plot([1.0, enr], [yi, yi], color=c, lw=3, zorder=2, solid_capstyle='round')
    ax.scatter(e['enrichment'], y, s=140, color=colors, edgecolor='0.25', lw=1, zorder=3)
    for yi, row in e.iterrows():
        xtext = max(row['enrichment'], 1.0)  # text always at the bar's right end
        ax.annotate(f"{row['enrichment']:.2f}×   peak {int(row['peak_nm'])} nm",
                    (xtext, yi), xytext=(10, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(e['trait'], fontsize=11)
    ax.set_xlim(0.4, vmax * 1.32)
    ax.set_ylim(-0.6, len(e) - 0.2)
    ax.set_xlabel('Enrichment  =  share of IG importance in known absorption regions  ÷  chance',
                  fontsize=9.5)
    ax.set_title('The 2D-CNN concentrates band importance in known absorption regions',
                 fontsize=12.5, fontweight='bold', loc='left')
    ax.set_title(f"{int((e['enrichment'] >= 1).sum())}/{len(e)} traits above chance",
                 fontsize=9.5, loc='right', color='0.45')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- Main figure: smoothed profiles + enrichment badges ------------------

def _smooth_segmented(wl, y, window=21, poly=3):
    """Savitzky-Golay smoothing applied per contiguous segment (never bridges
    the water-band gaps), to attenuate the band-flattening comb."""
    from scipy.signal import savgol_filter
    smoothed = y.astype(float).copy()
    bounds = [0, *(np.where(np.diff(wl) > 2)[0] + 1), len(wl)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a > window:
            smoothed[a:b] = savgol_filter(y[a:b], window, poly)
    return np.clip(smoothed, 0, None)


def plot_main_figure(wl, ig_mean, ig_std, df, out, smooth=31):
    """Publication main figure: smoothed per-trait IG profiles with absorption
    regions shaded, the peak marked, and a per-trait enrichment badge. Fuses the
    profile grid and the enrichment metric into a single figure."""
    from matplotlib.colors import TwoSlopeNorm
    enr = compute_enrichment(df).set_index('trait')
    norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    for ax, trait, m, s in zip(axes.ravel(), GHS_TRAIT_SHORT, ig_mean, ig_std):
        peak = m.max() + 1e-12
        ys = _smooth_segmented(wl, m / peak, smooth)
        ss = _smooth_segmented(wl, s / peak, smooth)
        _shade_context(ax, trait)
        wlb, yb = _broken(wl, ys)
        _, sb = _broken(wl, ss)
        ax.fill_between(wlb, np.clip(yb - sb, 0, None), yb + sb, color='tab:blue', alpha=0.2)
        ax.plot(wlb, yb, color='tab:blue', lw=1.4)
        ax.axvline(wl[ys.argmax()], color='tab:blue', ls=':', lw=0.9, alpha=0.6)
        e = float(enr.loc[trait, 'enrichment'])
        on_right = wl[ys.argmax()] < 1425  # peak on left half -> badge top-right
        ax.text(0.975 if on_right else 0.025, 0.93, f"{e:.2f}×", transform=ax.transAxes,
                ha='right' if on_right else 'left', va='top',
                fontsize=11, fontweight='bold', color='0.15',
                bbox=dict(boxstyle='round,pad=0.35', fc=plt.cm.RdYlGn(norm(e)), ec='0.3',
                          lw=0.7, alpha=0.92))
        ax.set_title(trait, fontsize=11, loc='left', fontweight='bold')
        ax.set_ylim(0, 1.16)
        ax.set_xlim(400, 2450)
    for ax in axes[-1]:
        ax.set_xlabel('Wavelength (nm)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Norm. IG importance')
    handles = [mpatches.Patch(color='tab:green', alpha=0.3, label='Known absorption region'),
               mpatches.Patch(color='0.85', label='Removed water band')]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle('Per-trait spectral band importance (Integrated Gradients, 3 seeds)',
                 y=0.978, fontsize=13)
    fig.text(0.5, 0.957, 'badge = enrichment of importance in known absorption regions vs chance',
             ha='center', fontsize=9, color='0.35', style='italic')
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- Conceptual pipeline figure ------------------------------------------

def plot_pipeline(wl, spectrum, img2d, cam_all, ig_all, trait_names, out):
    """Conceptual figure of the 1D->2D->CNN->1D pipeline for a few traits:
    (a) 1D spectrum -> (b) 2D Reshape image -> (c) Grad-CAM on the image ->
    (d) IG band importance unfolded back to the spectrum. Shows that the CNN
    attends to different image regions (hence spectral regions) per trait.
    """
    idx = [GHS_TRAIT_SHORT.index(t) for t in trait_names]
    fig = plt.figure(figsize=(16, 5.8))
    axbg = fig.add_axes([0, 0, 1, 1]); axbg.axis('off')
    axbg.set_xlim(0, 1); axbg.set_ylim(0, 1)

    ax_s = fig.add_axes([0.045, 0.20, 0.17, 0.58])
    wlb, sb = _broken(wl, spectrum)
    ax_s.plot(wlb, sb, color='0.2', lw=1.2)
    ax_s.set_title('(a) 1D spectrum', fontsize=11, loc='left')
    ax_s.set_xlabel('Wavelength (nm)', fontsize=8)
    ax_s.set_ylabel('Reflectance', fontsize=8)
    ax_s.tick_params(labelsize=7)

    ax_i = fig.add_axes([0.275, 0.28, 0.135, 0.44])
    ax_i.imshow(img2d, cmap='gray')
    ax_i.axis('off')
    ax_i.set_title('(b) 2D image (Reshape)', fontsize=11, loc='left')

    for row, (t, i) in enumerate(zip(trait_names, idx)):
        yb = 0.54 - row * 0.42
        ax_c = fig.add_axes([0.47, yb, 0.115, 0.34])
        _overlay_cam(ax_c, cam_all[i], img2d)
        ax_c.set_title(f'(c) Grad-CAM · {t}', fontsize=10, loc='left')

        ax_g = fig.add_axes([0.655, yb, 0.32, 0.34])
        y = _smooth_segmented(wl, ig_all[i] / (ig_all[i].max() + 1e-12))
        _shade_context(ax_g, t)
        wlb2, yb2 = _broken(wl, y)
        ax_g.plot(wlb2, yb2, color='tab:blue', lw=1.1)
        ax_g.set_xlim(400, 2450)
        ax_g.set_ylim(0, 1.12)
        ax_g.set_title(f'(d) IG importance · {t}', fontsize=10, loc='left')
        ax_g.tick_params(labelsize=7)
        ax_g.set_xlabel('Wavelength (nm)', fontsize=8) if row == len(trait_names) - 1 else None

    def arrow(x0, x1, y0, y1=None, label=None):
        y1 = y0 if y1 is None else y1
        axbg.annotate('', xy=(x1, y1), xytext=(x0, y0),
                      arrowprops=dict(arrowstyle='-|>', color='0.45', lw=2.2))
        if label:
            axbg.text((x0 + x1) / 2, max(y0, y1) + 0.04, label, ha='center',
                      fontsize=8.5, color='0.4', style='italic')
    arrow(0.225, 0.268, 0.50, label='reshape')
    arrow(0.420, 0.462, 0.50, label='2D-CNN')
    arrow(0.595, 0.648, 0.71, label='unfold / IG')
    arrow(0.595, 0.648, 0.29)
    fig.suptitle('From 1D spectrum to spectral band importance via 2D imaging',
                 fontsize=13, y=0.99)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--method', choices=['ig', 'gradcam', 'both'], default='both')
    ap.add_argument('--seeds', nargs='+', default=['', '_s240', '_s318'],
                    help="checkpoint dir suffixes (default: the 3 GHS seeds)")
    ap.add_argument('--exp-prefix', default='results/greenhs_reshape_efficientnet_b0')
    ap.add_argument('--n-samples', type=int, default=0, help='0 = full test set')
    ap.add_argument('--n-steps', type=int, default=50)
    ap.add_argument('--out', default='results/interpretation')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = wavelengths()
    device = args.device
    n_traits = len(GHS_TRAIT_SHORT)

    # Data: test spectra to attribute, train mean as IG baseline.
    train_s, _, test_s, test_y, _ = load_ghs_data(GHS_DATA_DIR)
    if args.n_samples and args.n_samples < len(test_s):
        sel = np.random.default_rng(0).choice(len(test_s), args.n_samples, replace=False)
        test_s, test_y = test_s[sel], test_y[sel]
    images = make_images(test_s, device)                       # (N,1,224,224)
    baseline_img = make_images(train_s.mean(0), device)        # (1,1,224,224)
    mean_img = images.mean(0)[0].numpy()
    # Attribute each trait only on samples where it is measured (mask out NaN).
    masks = [torch.from_numpy(~np.isnan(test_y[:, t])) for t in range(n_traits)]
    print(f"device={device}  images={tuple(images.shape)}  seeds={args.seeds}")
    print("  samples/trait: " + ", ".join(
        f"{tr}={int(m.sum())}" for tr, m in zip(GHS_TRAIT_SHORT, masks)))

    ig_seeds, ig_signed_seeds, cam_seeds = [], [], []
    for suf in args.seeds:
        exp = Path(args.exp_prefix + suf)
        if not (exp / 'fold_1' / 'best.ckpt').exists():
            print(f"  skip missing {exp}")
            continue
        print(f"[seed {suf or 's155'}] {exp}")
        model = load_model(exp, device)

        if args.method in ('ig', 'both'):
            imp = np.zeros((n_traits, N_BANDS))
            sig = np.zeros((n_traits, N_BANDS))
            for t in range(n_traits):
                imgs_t = images[masks[t]]
                attr = integrated_gradients(model, imgs_t, baseline_img, t,
                                            n_steps=args.n_steps, device=device)
                imp[t], sig[t] = attribution_to_bands(attr)
                if t == 0:
                    err = ig_completeness(model, imgs_t, baseline_img, 0, attr, device)
                    print(f"    IG completeness max err (trait 0): {err:.4f}")
            ig_seeds.append(imp)
            ig_signed_seeds.append(sig)

        if args.method in ('gradcam', 'both'):
            cams = np.stack([grad_cam(model, images[masks[t]], t, model.bn2,
                                      device=device).mean(0).numpy()
                             for t in range(n_traits)])
            cam_seeds.append(cams)

    # Aggregate over seeds and save.
    ig_mean = cam_mean = None
    if ig_seeds:
        ig_mean = np.mean(ig_seeds, 0)
        ig_std = np.std(ig_seeds, 0)
        ig_signed = np.mean(ig_signed_seeds, 0)
        df = pd.DataFrame({'band_index': np.arange(N_BANDS), 'wavelength_nm': wl})
        for t, tr in enumerate(GHS_TRAIT_SHORT):
            df[f'{tr}_imp_mean'] = ig_mean[t]
            df[f'{tr}_imp_std'] = ig_std[t]
            df[f'{tr}_signed_mean'] = ig_signed[t]
        df.to_csv(out / 'ig_importance_per_band.csv', index=False)
        plot_ig_grid(wl, ig_mean, ig_std, out / 'fig_ig_profiles_grid.pdf')
        compute_enrichment(df).to_csv(out / 'enrichment_summary.csv', index=False)
        plot_enrichment(df, out / 'fig_enrichment.pdf')
        plot_main_figure(wl, ig_mean, ig_std, df, out / 'fig_main.pdf')
        print("  saved CSVs + fig_ig_profiles_grid.pdf + fig_enrichment.pdf + fig_main.pdf")

    if cam_seeds:
        cam_mean = np.mean(cam_seeds, 0)
        np.save(out / 'gradcam_per_trait.npy', cam_mean)
        plot_gradcam_2d(cam_mean, mean_img, out / 'fig_gradcam_2d.pdf')
        print("  saved gradcam_per_trait.npy + fig_gradcam_2d.pdf")

    if ig_mean is not None and cam_mean is not None:
        plot_pipeline(wl, train_s.mean(0), baseline_img[0, 0].numpy(),
                      cam_mean, ig_mean, ['Cab', 'Cm'], out / 'fig_pipeline.pdf')
        for t, tr in enumerate(GHS_TRAIT_SHORT):
            plot_linking(t, wl, ig_mean[t], cam_mean[t], mean_img, out / f'fig_linking_{tr}.pdf')
        print(f"  saved fig_pipeline.pdf + {n_traits} linking figures")


if __name__ == '__main__':
    main()
