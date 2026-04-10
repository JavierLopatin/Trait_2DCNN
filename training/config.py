from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


TRAIT_NAMES = [
    'Anthocyanin content (μg/cm²)',
    'Boron content (mg/cm²)',
    'C content (mg/cm²)',
    'Ca content (mg/cm²)',
    'Carotenoid content (μg/cm²)',
    'Cellulose (mg/cm²)',
    'Chl content (μg/cm²)',
    'Copper content (mg/cm²)',
    'EWT (mg/cm²)',
    'Fiber (mg/cm²)',
    'LAI (m²/m²)',
    'LMA (g/m²)',
    'Lignin (mg/cm²)',
    'Magnesium content (mg/cm²)',
    'Manganese content (mg/cm²)',
    'N content (mg/cm²)',
    'NSC (mg/cm²)',
    'Phosphorus content (mg/cm²)',
    'Potassium content (mg/cm²)',
    'Sulfur content (mg/cm²)',
]

TRAIT_SHORT = [
    'Anth', 'B', 'C', 'Ca', 'Car', 'Cel', 'Chl', 'Cu', 'EWT', 'Fiber',
    'LAI', 'LMA', 'Lignin', 'Mg', 'Mn', 'N', 'NSC', 'P', 'K', 'S',
]

N_TRAITS = 20
N_BANDS = 1721
BAND_START = '400'
BAND_END = '2450'


@dataclass
class TrainConfig:
    # Paths
    data_dir: Path = Path('multi-traitretrieval/dataset')
    output_dir: Path = Path('results')

    # Transform
    transform_name: str = 'cwt'
    output_size: int = 224

    # Model
    model_name: str = 'efficientnet_b0'
    pretrained: bool = False

    # Training
    n_folds: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    val_split: float = 0.15
    num_workers: int = 4
    seed: int = 42

    # Augmentation (Cherif-style)
    aug_baseline_shift: float = 0.02
    aug_multiplicative: float = 0.02

    # Device
    accelerator: str = 'auto'
    devices: int = 1

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
