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

- **2D-CNN**: EfficientNet-B0/V2, ResNet-50, ConvNeXt
- **Transformers**: ViT, Swin Transformer
- **Hybrid**: CNN backbone + Transformer head
- **Baseline (existing)**: 1D-CNN EfficientNet-B0, PLSR

## Project Structure

```
Trait_2DCNN/
├── data/
│   └── raw/                             # 42-dataset spectral database
├── multi-traitretrieval/                # Original 1D pipeline (Cherif et al.)
│   ├── data_module_F.py                 # Data loading, trait definitions
│   ├── feature_module_F.py              # Spectral preprocessing, Savitzky-Golay
│   ├── model_module_F.py                # Training utilities, augmentation
│   ├── evaluation_module_F.py           # Metrics, visualization
│   ├── model_builder.py                 # 1D-CNN architecture
│   ├── EfficientNet1D_builder.py        # EfficientNet-B0 adapted to 1D
│   ├── mainTrain.py                     # Main training script
│   ├── dataset/                         # 5-fold CV splits
│   ├── models/                          # Trained 1D models
│   └── Predictions/                     # Hyperspectral image inference
├── transforms/                          # 1D→2D transformation modules
│   ├── base.py
│   ├── cwt_transform.py
│   ├── cos2d_transform.py
│   ├── reshape_transform.py
│   ├── gaf_transform.py
│   ├── spectrogram_transform.py
│   └── deepinsight_transform.py
├── models/                              # 2D model architectures
├── training/                            # Training pipeline
├── evaluation/                          # Method comparison
├── notebooks/                           # Experiments
├── CLAUDE.md                            # Implementation plan
├── requirements.txt
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

## Evaluation

All methods are evaluated using the same 5-fold cross-validation splits as Cherif et al. (2023) for fair comparison:

- **Metrics**: R², RMSE, nRMSE, MAE, Bias
- **Baselines**: PLSR, 1D-CNN (single-trait), 1D-CNN (multi-trait)
- **Training**: Weakly supervised multi-trait regression (20 traits simultaneously)

## References

- Cherif et al. (2023). *From spectra to plant functional traits: Transferable multi-trait models from heterogeneous and sparse data.* Remote Sensing of Environment. [DOI: 10.1016/j.rse.2023.113580](https://doi.org/10.1016/j.rse.2023.113580)
- Shuai et al. (2025). *Multi-channel spectrogram + ConvNeXt for Vis-NIR soil prediction.* J. Chemometrics.
- Mokari et al. (2025). *Spider plot transformation for spectral deep learning.* Advanced Intelligent Systems.
- Hennessy et al. (2022). *Reshaping hyperspectral data into 2D for CNN.* Remote Sensing.
- Contreras (2025). *XAI-2DCOS: 2D correlation spectroscopy + deep learning.* J. Chemometrics.

## License

MIT
