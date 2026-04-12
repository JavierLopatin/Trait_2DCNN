# Trait_2DCNN

## Project Overview

Extension of Cherif et al. (2023) multi-trait retrieval pipeline. The goal is to transform 1D hyperspectral reflectance signals into 2D image representations, then apply 2D-CNN and Transformer architectures for simultaneous regression of 20 plant functional traits.

## Repository

- **GitHub**: https://github.com/JavierLopatin/Trait_2DCNN (private)
- **Cherif baseline**: `multi-traitretrieval/` (gitignored, separate clone)

## Existing Codebase (Cherif et al. 2023)

- **Original repo**: `multi-traitretrieval/`
- **Current model**: 1D-CNN based on EfficientNet-B0 (`EfficientNet1D_builder.py`)
- **Modules**: `data_module_F.py`, `feature_module_F.py`, `model_module_F.py`, `evaluation_module_F.py`
- **Training**: 5-fold CV, weakly supervised learning (incomplete, inexact, incompleteTRY)
- **Baseline**: PLSR models for comparison
- **Traits (20)**: LMA, LAI, N, C, Chl, Car, Anth, EWT, Cellulose, Lignin, Fiber, NSC, P, Ca, Mg, K, S, B, Cu, Mn (all area-basis)

## Data Strategy

### Source: Cherif's preprocessed CV splits (`multi-traitretrieval/dataset/`)

Use Cherif's splits directly for 1:1 comparison with the original pipeline.

| File | Rows | Description |
|------|------|-------------|
| `fillCV_{1-5}.csv` | ~12,200 | Train splits (upsampled to balance datasets) |
| `testCV_{1-5}.csv` | ~1,115 | Test splits |
| `samp_w_tr_{1-5}.csv` | ~12,200 | Sample weights per fold |

**Structure of each CSV**: 20 trait columns + 1721 spectral bands (400-2450nm)

### Data Characteristics
- **50 datasets** from 28 studies, diverse ecosystems and sensors
- **Sparse**: No samples have all 20 traits complete. Average ~7 traits/sample
- **Upsampled**: Small datasets are oversampled to balance
- **NaN = missing data**: Handled with masked loss (ignore NaN in the loss function)

### Why This Data and Not Raw (`data/raw/`)
- The `fillCV` files are already **clean**: only 20 traits + 1721 bands, no metadata
- **Upsampled** and balanced by dataset
- **Identical 5-fold CV** to Cherif → direct comparison
- Avoids regenerating splits and preprocessing
- Gap-filling (`CNNmultiInexact`) happens during training, not in the CSVs

### Data Pipeline
```
fillCV_*.csv → extract spectra (cols "400" to "2450", 1721 bands)
            → transform 1D→2D (CWT, 2D-COS, etc.)
            → train 2D model with masked loss
testCV_*.csv → same transformation → evaluate with Cherif's metrics
```

### Validation and Metrics
- **Same 5 CV folds** as Cherif (splits already defined)
- **Masked loss**: Ignores NaN in the loss function (already implemented in `model_builder.py`: `MaskedRmse`, `MaskedR2`)
- **Scaler**: PowerTransformer Box-Cox for traits (reuse `save_scaler()` from `model_module_F.py`)
- **Data augmentation**: Baseline shift + multiplicative (reuse from `model_module_F.py`)
- **Metrics**: R², RMSE, nRMSE, MAE, Bias (reuse `all_scores()` from `model_module_F.py`)
- **Comparison baselines**: PLSR, 1D-CNN single-trait, 1D-CNN multi-trait (results already in `multi-traitretrieval/models/`)

## 1D-to-2D Transformations

### Implemented Transforms

| Method | Channels | Status | Mean R² | Description |
|--------|----------|--------|---------|-------------|
| **Reshape** | 1 | Done | **0.580** | Direct reshape baseline (best overall) |
| **Spectrogram** | 3 | Done | **0.576** | STFT with 3 windows as RGB (best on pigments) |
| **CWT** | 1 | Done | 0.542 | Continuous Wavelet scalograms |
| **COS2D** | 2 | Done | 0.541 | Synchronous + asynchronous correlation maps |
| **GAF** | 2 | Done | 0.536 | Gramian Angular Fields (GASF + GADF) |
| **MTF** | 1 | Done | 0.358 | Markov Transition Field (weak standalone) |
| reshape+cwt | 2 | Done | 0.541 | Composite: no improvement over singles |
| reshape+cwt+spectro | 5 | Done | 0.537 | Composite: no improvement over singles |

### Not Implemented (Low Priority)

- **Recurrence Plot (RP)** — Similar to COS2D (pairwise distances vs correlations), likely redundant
- **WPGA** — Extends GAF which is weakest performer; multi-scale unlikely to rescue it
- **Spider Plot** — Polar rendering, only useful for ImageNet pretraining
- **DeepInsight / IGTD** — Complex feature embedding, deferred

### Multi-Channel Composite Transforms

The `CompositeTransform` stacks multiple transforms as channels. Use `+` syntax:

```bash
# 2-channel: reshape + cwt
python -m training.train_2d --transform reshape+cwt --model efficientnet_b0

# 5-channel: reshape + cwt + spectrogram
python -m training.train_2d --transform reshape+cwt+spectrogram --model efficientnet_b0
```

Complementarity analysis showed high correlation between all transforms (r>0.93).
Composite experiments confirmed that **multi-channel does not improve** over single transforms:

| Composite | Channels | Mean R² | vs Reshape (0.580) |
|-----------|----------|---------|---------------------|
| reshape+cwt | 2 | 0.541 | -0.039 |
| reshape+cwt+spectrogram | 5 | 0.537 | -0.043 |

The model struggles to learn cross-channel relationships when channels encode
fundamentally different representations. Simpler single-transform inputs work better.

### Experimental Results vs Cherif 1D-CNN Baseline

**Overall paired t-test (mean R² across 20 traits, 5-fold CV, α=0.05):**

| Method | Cherif R² | Ours R² | Delta | p-value | Sig. |
|--------|-----------|---------|-------|---------|------|
| Reshape | 0.495 | 0.580 | +0.085 | 0.004 | Yes |
| Spectrogram | 0.495 | 0.576 | +0.081 | 0.012 | Yes |
| COS2D | 0.495 | 0.541 | +0.046 | 0.019 | Yes |
| CWT | 0.495 | 0.542 | +0.047 | 0.036 | Yes |
| GAF | 0.495 | 0.536 | +0.041 | 0.102 | No |

All methods except GAF significantly outperform Cherif (p<0.05). No method is significantly worse on any individual trait.

**Per-trait highlights:**
- Reshape: 11/20 traits significantly better, best on structural/chemical (Cel, Lignin, Fiber, EWT)
- Spectrogram: 10/20 significantly better, best on pigments (Car, Chl, Anth, LAI)
- Hardest traits (Ca, Mg, K, Mn): all methods improve but margins are small

### Model Architectures

All via `timm` with `create_model()`. Tested with Reshape transform:

| Architecture | Type | Params | Mean R² | Notes |
|-------------|------|--------|---------|-------|
| **EfficientNet-B0** | CNN | ~5M | **0.580** | Best on all 20 traits |
| ResNet-50 | CNN | ~25M | 0.551 | Classical, decent but worse |
| ConvNeXt-Tiny | CNN | ~28M | 0.482 | Underperforms despite modern design |
| Swin-Tiny | Transformer | ~28M | -0.037 | Failed — too few samples, no pretraining |

EfficientNet-B0 dominates despite being the smallest model (~5M params).
Larger models overfit with ~12K training samples. Transformers (Swin) fail
completely without pretraining on this dataset size.

### Training & Evaluation

- **5-fold CV** using Cherif's `fillCV_*.csv` / `testCV_*.csv` splits
- **Masked loss** (MaskedMSELoss) for NaN handling — 20-trait simultaneous regression
- **PowerTransformer** (Yeo-Johnson) scaler for trait labels
- **Data augmentation**: baseline shift ±2% + multiplicative ±2%
- **Metrics**: R², RMSE, nRMSE, MAE, Bias
- **Early stopping**: patience=15 on validation loss
- **Optimizer**: AdamW with cosine annealing LR

## Environment

- **Conda env**: `/mnt/rapidita_4T/envs/trait2dcnn` (Python 3.11)
- **Framework**: PyTorch 2.6.0 + Lightning 2.6.1 (new pipeline, instead of TensorFlow)
- **GPU**: 2x RTX 2060 (6 GB each) — CUDA 12.4 + cuDNN 9.1.0 working
- **Activate**: `conda activate /mnt/rapidita_4T/envs/trait2dcnn`

### Key Dependencies

```
torch, torchvision, torchaudio  # PyTorch + CUDA 12.4
timm                            # Pretrained models (ViT, Swin, EfficientNet, ConvNeXt)
lightning, torchmetrics          # Training framework
pywt                            # CWT, wavelet packet decomposition
pyts                            # GAF, MTF, Recurrence Plots
ssqueezepy                      # Synchrosqueezed transforms
librosa                         # STFT spectrograms
pandas, scikit-learn, scipy, matplotlib, seaborn  # Science stack
```

## Implementation Details

### Architecture Decisions
- **PyTorch + Lightning** instead of TensorFlow (better ecosystem for Transformers via `timm`)
- **Pre-computed transforms** saved as memory-mapped arrays in `cache/` (avoids recomputing each epoch)
- **Image size 224x224** for GPU training (64x64 available for quick CPU tests)
- **Models via `timm`**: A single `create_model()` for all architectures
- **Composite transforms**: `CompositeTransform` stacks sub-transform caches as channels (no re-precomputation needed)

### Directory Structure

```
Trait_2DCNN/
├── data/raw/                        # Raw spectral database (7,897 samples)
├── multi-traitretrieval/            # Cherif et al. 1D pipeline
│   ├── dataset/                     # CV splits: fillCV_*.csv, testCV_*.csv (INPUT DATA)
│   └── models/multi_{1-5}/inexact/  # Cherif predictions + observations (for comparison)
├── transforms/                      # 1D→2D transformation modules
│   ├── base.py                      # Abstract base class
│   ├── cwt_transform.py             # Continuous Wavelet Transform (1ch scalogram)
│   ├── cos2d_transform.py           # 2D Correlation Spectroscopy (2ch: sync + async)
│   ├── reshape_transform.py         # Direct reshape baseline (1ch)
│   ├── gaf_transform.py             # Gramian Angular Fields (2ch: GASF + GADF)
│   ├── spectrogram_transform.py     # Multi-channel STFT (3ch: 3 windows → RGB)
│   ├── mtf_transform.py             # Markov Transition Field (1ch)
│   └── composite_transform.py       # Multi-channel stacking of any transforms
├── models/
│   └── trait_model.py               # Unified model factory via timm
├── training/
│   ├── config.py                    # TrainConfig dataclass + trait definitions
│   ├── data_loader.py               # Cached + on-the-fly datasets, composite cache stacking
│   ├── precompute.py                # Pre-compute transforms to disk (memmap)
│   ├── losses.py                    # MaskedMSE, MaskedHuber, MaskedR², MaskedRMSE
│   ├── lightning_module.py          # TraitRegressionModule (Lightning)
│   └── train_2d.py                  # Main training script (CLI, --start-fold, composite)
├── evaluation/
│   ├── compare_methods.py           # Load baselines + 2D results, generate plots
│   └── statistical_comparison.py    # Paired t-tests vs Cherif, multi-channel analysis
├── cache/                           # Pre-computed transform images (memmap, gitignored)
├── results/                         # Experiment outputs (metrics, checkpoints)
├── environment.yml                  # Conda environment spec
├── CLAUDE.md
└── README.md
```

### Running experiments

```bash
# 1. Pre-compute transforms (224x224 for GPU)
python -m training.precompute --transform reshape --output-size 224
python -m training.precompute --transform cwt --output-size 224
python -m training.precompute --transform mtf --output-size 224

# 2. Train single transforms
python -m training.train_2d --transform reshape --model efficientnet_b0 --epochs 100
python -m training.train_2d --transform cwt --model efficientnet_b0 --epochs 100

# 3. Train multi-channel composite (reuses existing caches)
python -m training.train_2d --transform reshape+cwt --model efficientnet_b0 --epochs 100
python -m training.train_2d --transform reshape+cwt+spectrogram --model efficientnet_b0

# 4. Resume from a specific fold
python -m training.train_2d --transform reshape --model efficientnet_b0 --start-fold 3

# 5. Statistical comparison
python -m evaluation.statistical_comparison

# 6. Compare results
python -m evaluation.compare_methods --results-dir results/
```

## References

- Cherif et al. (2023) — From spectra to plant functional traits (base paper)
- Shuai et al. (2025) — Multi-channel spectrogram + ConvNeXt for Vis-NIR soil prediction
- Mokari et al. (2025) — Spider plot transformation for spectral deep learning
- Hennessy et al. (2022) — Direct reshaping of hyperspectral data for 2D-CNN
- Deev et al. (2024) — Spectrum is a Picture: 2D-CNN in spectral processing
- Contreras (2025) — XAI-2DCOS: 2D correlation spectroscopy + deep learning
- Sharma et al. (2019) — DeepInsight: tabular to image for CNN
- Zhu et al. (2021) — IGTD: tabular data to images

## NotebookLM

- Connected via `notebooklm-py` (authenticated as javierlopatin@gmail.com)
- Notebook "Cherif2023" (ID: `01abac6a-3fac-450c-a22a-67a1f49f769d`) contains the base paper PDF
