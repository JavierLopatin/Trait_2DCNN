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

| Method | Description | Output | Library |
|--------|-------------|--------|---------|
| **CWT** | Continuous Wavelet Transform scalograms | (scales, bands) | `pywt` |
| **2D-COS** | Synchronous/asynchronous correlation spectroscopy | (bands, bands) | `numpy` |
| **Direct Reshape** | Simple reshape to square matrix (baseline) | (h, w) | `numpy` |
| **GAF** | Gramian Angular Summation/Difference Fields | (n, n) | `pyts` |
| **Multi-Ch Spectrogram** | STFT with 3 window functions as RGB channels | (freq, time, 3) | `scipy` |
| **DeepInsight / IGTD** | Feature embedding placing correlated bands nearby | (H, W) | `pyDeepInsight`, `TINTOlib` |
| **Spider Plot** | Polar rendering for pretrained model compatibility | (224, 224, 3) | `matplotlib` |
| **MTF** | Markov Transition Field | (n, n) | `pyts` |
| **Recurrence Plot** | Pairwise distance matrix | (n, n) | `pyts` |
| **WPGA** | Wavelet Packet decomposition + GAF | multi-channel | `pywt` + `pyts` |

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

Building on Case Study 1, we investigate whether 2D representations unlock vision-domain capabilities impossible with 1D spectra, using the [GreenHyperSpectra](https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra) dataset (Cherif et al., 2025, NeurIPS):

| # | Pretraining | Fine-tuning | Test | Question |
|---|------------|-------------|------|----------|
| 1 | — | Real labeled (4,508) | Real (1,127) | Supervised 2D baseline |
| 3 | MAE-2D (139K unlabeled) | Real labeled | Real | Does 2D pretraining beat 1D? |
| 5 | MAE-2D (139K unlabeled) | PROSAIL synthetic | Real | Zero real labels needed? |

Comparison baselines: Cherif et al. (2025) supervised (R² 0.587) and MAE-1D Fine-Tuned (R² 0.645).

## Project Structure

```
Trait_2DCNN/
├── data/raw/                            # Raw spectral database
├── multi-traitretrieval/                # Cherif et al. 1D pipeline (gitignored)
├── transforms/                          # 1D→2D transformation modules
│   ├── base.py                          # Abstract base class
│   ├── reshape_transform.py             # Direct reshape (1ch)
│   ├── cwt_transform.py                 # Continuous Wavelet Transform (1ch)
│   ├── cos2d_transform.py               # 2D Correlation Spectroscopy (2ch)
│   ├── gaf_transform.py                 # Gramian Angular Fields (2ch)
│   ├── spectrogram_transform.py         # Multi-channel STFT (3ch)
│   ├── mtf_transform.py                 # Markov Transition Field (1ch)
│   └── composite_transform.py           # Multi-channel stacking
├── models/
│   └── trait_model.py                   # Unified model factory via timm
├── training/
│   ├── config.py                        # TrainConfig + trait definitions
│   ├── data_loader.py                   # Cached + on-the-fly datasets
│   ├── precompute.py                    # Pre-compute transforms to disk
│   ├── losses.py                        # Masked losses for NaN handling
│   ├── lightning_module.py              # Lightning training module
│   └── train_2d.py                      # Main CLI script
├── evaluation/
│   ├── compare_methods.py               # Generate comparison plots
│   └── statistical_comparison.py        # Paired t-tests vs Cherif
├── data/
│   ├── GreenHyperSpectra/               # Cherif 2025 dataset (gitignored)
│   └── papers/                          # Reference papers
├── prosail/                             # PROSAIL LUT generation (R + Python)
├── models/
│   ├── trait_model.py                   # Unified model factory via timm
│   └── mae_2d.py                       # 2D Masked Autoencoder (ViT-based)
├── cache/                               # Pre-computed transforms (gitignored)
├── results/                             # Experiment outputs (gitignored)
├── CLAUDE.md                            # Detailed implementation notes
└── README.md
```

## Setup

```bash
# Create environment
conda create -n trait2dcnn python=3.11
conda activate trait2dcnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# 1. Pre-compute transforms
python -m training.precompute --transform reshape --output-size 224

# 2. Train
python -m training.train_2d --transform reshape --model efficientnet_b0 --epochs 100

# 3. Multi-channel composite (reuses existing caches)
python -m training.train_2d --transform reshape+cwt --model efficientnet_b0

# 4. Statistical comparison vs Cherif
python -m evaluation.statistical_comparison
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
