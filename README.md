# Trait_2DCNN: Plant Functional Traits from 2D Spectral Representations

Retrieval of plant functional traits from hyperspectral canopy reflectance using **2D image representations** of 1D spectra, combined with 2D-CNN and Transformer architectures for multi-trait regression.

Extends the work of [Cherif et al. (2023)](https://doi.org/10.1016/j.rse.2023.113580) — *"From spectra to plant functional traits: Transferable multi-trait models from heterogeneous and sparse data"* — by exploring whether transforming spectral signals into 2D images improves trait prediction over the original 1D-CNN approach.

## Motivation

The original pipeline uses a 1D-CNN (EfficientNet-B0 adapted to 1D) to predict 20 plant traits simultaneously from hyperspectral reflectance. This project investigates whether **2D representations** of spectral data can capture inter-band relationships and multi-scale patterns that 1D convolutions miss, leveraging mature 2D architectures (EfficientNet, ResNet, ViT, Swin Transformer).

## Data

- **42 datasets** from 28 studies across diverse ecosystems (forests, grasslands, tundra, shrublands, croplands)
- **5,573 canopy reflectance spectra** (400-2500 nm, 1721 bands after water absorption removal)
- Sensors: ASD FieldSpec, AVIRIS, NEON AOP, and others
- Geographic coverage: Americas, Asia, Europe

### Target Traits (20, area-basis)

| Category | Traits |
|----------|--------|
| Structural | LMA, LAI |
| Pigments | Chlorophyll, Carotenoids, Anthocyanins |
| Chemical | Carbon, Nitrogen, Phosphorus, EWT, Cellulose, Lignin, Fiber, NSC |
| Minerals | Ca, Mg, K, S, B, Cu, Mn |

## 1D-to-2D Transformation Methods

Nine transforms are implemented (registered in `transforms/__init__.py`). All output
`224 × 224` images. The first three are locality-preserving layouts of the raw spectrum;
the rest encode complementary time–frequency or inter-band information.

| Method (`key`) | Ch | Description | Library |
|----------------|----|-------------|---------|
| **Reshape** (`reshape`) | 1 | Direct reshape to a square matrix (baseline, best overall) | `numpy` |
| **Serpentine** (`serpentine`) | 1 | Boustrophedon reshape — alternating row direction (preserves band adjacency across rows) | `numpy` |
| **Hilbert** (`hilbert`) | 1 | Reshape along a Hilbert space-filling curve (preserves 1D locality in 2D) | `numpy` |
| **CWT** (`cwt`) | 1 | Continuous Wavelet Transform scalograms | `pywt` |
| **Spectrogram** (`spectrogram`) | 3 | STFT with 3 window functions as RGB channels | `scipy` |
| **2D-COS** (`cos2d`) | 2 | Synchronous/asynchronous correlation spectroscopy | `numpy` |
| **GAF** (`gaf`) | 2 | Gramian Angular Summation/Difference Fields | `numpy` |
| **NDI** (`ndi`) | 3 | All-band normalized-difference matrix `(Rᵢ−Rⱼ)/(Rᵢ+Rⱼ)`, `|Rᵢ−Rⱼ|`, `√(RᵢRⱼ)` (all-pairs vegetation-index principle) | `numpy` |
| **MTF** (`mtf`) | 1 | Markov Transition Field | `pyts` |

Composites stack any transforms as channels via the `+` syntax (e.g. `reshape+cwt+ndi`).

## Model Architectures

Tested with Reshape transform (best performing):

| Architecture | Type | Params | Mean R² | Notes |
|-------------|------|--------|---------|-------|
| **EfficientNet-B0** | CNN | ~5M | **0.580** | Best on all 20 traits |
| ResNet-50 | CNN | ~25M | 0.551 | Classical, decent |
| ConvNeXt-Tiny | CNN | ~28M | 0.482 | Underperforms |
| Swin-Tiny | Transformer | ~28M | -0.037 | Failed (insufficient data, no pretraining) |

**Baseline**: Cherif et al. 1D-CNN (CNNmultiIncomplete) = 0.495 mean R²

## Results

### Transform Comparison (EfficientNet-B0, 224x224, 5-fold CV)

| Transform | Channels | Mean R² | vs Cherif (p-value) |
|-----------|----------|---------|---------------------|
| **Reshape** | 1 | **0.580** | +0.085 (p=0.004) |
| Spectrogram | 3 | 0.576 | +0.081 (p=0.012) |
| CWT | 1 | 0.542 | +0.047 (p=0.036) |
| COS2D | 2 | 0.541 | +0.046 (p=0.019) |
| GAF | 2 | 0.536 | +0.041 (p=0.102) |
| MTF | 1 | 0.358 | — |

All methods except GAF and MTF significantly outperform Cherif's 1D-CNN (paired t-test, p<0.05). Multi-channel composites (reshape+cwt, reshape+cwt+spectrogram) did not improve over single transforms.

## Case Study 2: Pretraining & PROSAIL Domain Transfer

Using [GreenHyperSpectra](https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra) (Cherif et al., 2025, NeurIPS), we test whether 2D representations unlock pretraining capabilities impossible with 1D spectra:

| # | Pretraining | Fine-tuning | Test | R² | Cherif 1D |
|---|------------|-------------|------|-----|-----------|
| 1 | — | Real (4,508) | Real (1,127) | **0.684 ± 0.001** | 0.587 |
| 3 | MAE-2D (139K) | Real | Real | **0.667 ± 0.006** | 0.645 (MAE-1D) |
| 5 | MAE-2D (139K) | PROSAIL-PRO (50K) | Real | -0.66 | — |

All 2D methods outperform their 1D counterparts. MAE-2D pretraining on 139K unlabeled images surpasses Cherif's MAE-1D (+0.022), confirming the advantage of 2D representations for self-supervised learning on spectral data. Direct domain transfer from PROSAIL simulations to real data fails (R² < 0), consistent with literature findings on the persistent gap between RTM-simulated and observed spectra (Mederer et al., 2025).

### Transform comparison on GHS (in-distribution, 8 traits, 3 seeds)

Significance is a Wilcoxon signed-rank test paired across the 8 traits vs the **reported** Cherif 2025 supervised 1D baseline (R² 0.587):

| Transform | Ch | Mean R² | Δ vs 1D | Wilcoxon p |
|-----------|----|---------|---------|-----------|
| **Reshape** | 1 | **0.684 ± 0.001** | +0.097 | 0.008 * |
| Serpentine | 1 | 0.675 ± 0.004 | +0.089 | 0.008 * |
| Hilbert | 1 | 0.674 ± 0.005 | +0.087 | 0.008 * |
| CWT | 1 | 0.641 ± 0.004 | +0.054 | 0.016 * |
| Spectrogram | 3 | 0.636 ± 0.025 | +0.049 | 0.008 * |
| Reshape+CWT+NDI (composite) | 5 | 0.666 ± 0.017 | +0.079 | 0.008 * |
| NDI | 3 | 0.630 ± 0.029 | +0.043 | 0.055 |
| GAF | 2 | 0.609 | +0.023 | 0.250 |
| COS2D | 2 | 0.556 | −0.031 | 0.195 |
| MTF | 1 | 0.419 | −0.168 | 0.008 * |

Reshape remains the best and simplest 2D encoding. The locality-preserving reorderings (Serpentine, Hilbert) match it; the all-pairs NDI and multi-channel composites do **not** beat it, confirming that the 2D-CNN already exploits the relevant inter-band structure from the single raw layout. Our supervised Reshape-2D and MAE-2D fine-tuning both **significantly** beat their Cherif 1D counterparts (p = 0.008, 0.039).

### Out-of-distribution (cross-dataset) generalization

Replicating the Cherif 2025 OOD protocol 1:1 (leave-5-datasets-out cross-validation over the 50 source datasets, Box-Cox scaling, masked-Huber loss, sub-sampled macro metric), swapping only the model:

| Method (OOD) | R² | Cherif 1D counterpart | Wilcoxon p |
|--------------|-----|-----------------------|-----------|
| Reshape-2D (supervised) | 0.273 | 0.243 (supervised) | 0.46 (n.s.) |
| MAE-2D-FT | 0.273 | 0.311 (MAE-FR-FT) | 0.20 (n.s.) |

Under cross-dataset shift the 2D advantage does not carry over: the supervised 2D edge over the 1D baseline is not significant, and 2D MAE pretraining (unlike in-distribution) does not beat Cherif's best 1D MAE. Both models drop sharply out of distribution.

### Spectral variable importance (interpretation)

Per-band importance of the best model (Reshape + EfficientNet-B0) is recovered with Integrated Gradients and compared against a continuous **PROSPECT-PRO/4SAIL theoretical sensitivity** (each trait's parameter swept while others are held fixed). Agreement is positive but modest and trait-dependent (Pearson r up to 0.45 for protein/water), grounding the learned band selection in radiative transfer rather than hand-drawn absorption regions.

## Project Structure

```
Trait_2DCNN/
├── multi-traitretrieval/                # Cherif et al. 2023 1D pipeline (gitignored)
├── transforms/                          # 1D→2D transformation modules
│   ├── base.py                          # Abstract base class
│   ├── reshape_transform.py             # Direct reshape (1ch)
│   ├── serpentine_transform.py          # Boustrophedon reshape (1ch)
│   ├── hilbert_transform.py             # Hilbert-curve reshape (1ch)
│   ├── ndi_transform.py                 # All-band normalized-difference matrix (3ch)
│   ├── cwt_transform.py                 # Continuous Wavelet Transform (1ch)
│   ├── cos2d_transform.py               # 2D Correlation Spectroscopy (2ch)
│   ├── gaf_transform.py                 # Gramian Angular Fields (2ch)
│   ├── spectrogram_transform.py         # Multi-channel STFT (3ch)
│   ├── mtf_transform.py                 # Markov Transition Field (1ch)
│   └── composite_transform.py           # Multi-channel stacking (`+` syntax)
├── models/
│   ├── trait_model.py                   # Unified model factory via timm
│   └── mae_2d.py                        # 2D Masked Autoencoder (ViT-based)
├── training/
│   ├── config.py                        # TrainConfig + trait definitions
│   ├── data_loader.py                   # Cached + on-the-fly datasets
│   ├── precompute.py                    # Pre-compute transforms to disk
│   ├── losses.py                        # Masked losses for NaN handling
│   ├── lightning_module.py              # Lightning training module
│   ├── train_2d.py                      # Main supervised CLI
│   ├── pretrain_mae.py / finetune_mae.py  # MAE-2D pretraining + fine-tuning
│   ├── ood_cherif2025.py                # Ported Cherif 2025 OOD protocol (attributed)
│   └── train_ood.py                     # Cross-dataset OOD driver
├── evaluation/
│   ├── compare_methods.py               # Comparison plots
│   ├── statistical_comparison.py        # Paired tests vs Cherif (Case Study 1)
│   ├── significance_cherif2025.py       # Wilcoxon vs Cherif 2025 (ID + OOD)
│   └── spectral_importance.py           # IG / Grad-CAM + PROSAIL sensitivity
├── prosail/                             # PROSAIL LUT + sensitivity (R + Python)
├── data/                                # Spectral datasets (gitignored)
├── cache/                               # Pre-computed transforms (gitignored)
├── results/                             # Experiment outputs
└── README.md
```

## Setup

```bash
# Create environment from spec
conda env create -f environment.yml
conda activate trait2dcnn
```

## Usage

```bash
# 1. Pre-compute transforms (GHS shown; --dataset cherif2023 for the 20-trait 5-fold CV)
python -m training.precompute --transform reshape --output-size 224 --dataset greenhs

# 2. Train (single transform or composite via `+`)
python -m training.train_2d --transform reshape --model efficientnet_b0 --dataset greenhs --seed 155
python -m training.train_2d --transform reshape+cwt+ndi --model efficientnet_b0 --dataset greenhs

# 3. MAE-2D pretraining + fine-tuning
python -m training.pretrain_mae --epochs 300
python -m training.finetune_mae --checkpoint results/mae_pretrained/best.pt --dataset greenhs

# 4. Out-of-distribution (cross-dataset) evaluation, Cherif 2025 protocol
CUDA_VISIBLE_DEVICES=0 python -m training.train_ood --model efficientnet_b0   # supervised 2D
CUDA_VISIBLE_DEVICES=0 python -m training.train_ood --model mae               # MAE-2D fine-tuning

# 5. Significance vs Cherif 2025 (Wilcoxon paired across traits, ID + OOD)
python -m evaluation.significance_cherif2025

# 6. Spectral variable importance (IG / Grad-CAM + PROSAIL sensitivity)
python -m evaluation.spectral_importance --method both
```

## Evaluation

All methods are evaluated using the same 5-fold cross-validation splits as Cherif et al. (2023) for fair comparison:

- **Metrics**: R², RMSE, nRMSE, MAE, Bias
- **Baselines**: PLSR, 1D-CNN (single-trait), 1D-CNN (multi-trait)
- **Training**: Weakly supervised multi-trait regression (20 traits simultaneously)
- **Statistical tests**: Paired t-tests across folds (α=0.05)

## References

- Cherif et al. (2023). *From spectra to plant functional traits: Transferable multi-trait models from heterogeneous and sparse data.* Remote Sensing of Environment. [DOI: 10.1016/j.rse.2023.113580](https://doi.org/10.1016/j.rse.2023.113580)
- Cherif et al. (2025). *GreenHyperSpectra: A multi-source hyperspectral dataset for global vegetation trait prediction.* NeurIPS 2025. [arXiv: 2507.06806](https://arxiv.org/abs/2507.06806)
- Shuai et al. (2025). *Multi-channel spectrogram + ConvNeXt for Vis-NIR soil prediction.* J. Chemometrics.
- Mokari et al. (2025). *Spider plot transformation for spectral deep learning.* Advanced Intelligent Systems.
- Hennessy et al. (2022). *Reshaping hyperspectral data into 2D for CNN.* Remote Sensing.
- Contreras (2025). *XAI-2DCOS: 2D correlation spectroscopy + deep learning.* J. Chemometrics.

## License

MIT
