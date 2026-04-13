from dataclasses import dataclass
from pathlib import Path


# --- Dataset identifiers (avoid raw string comparisons) ---
DATASET_CHERIF = 'cherif2023'
DATASET_GHS = 'greenhs'
DATASET_GHS_UNLABELED = 'greenhs_unlabeled'
DATASET_PROSAIL = 'prosail'

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

# GreenHyperSpectra (Cherif et al. 2025) -- 8 PROSAIL-related traits
GHS_TRAIT_NAMES = ['cab', 'car', 'anth', 'cw', 'cm', 'LAI', 'cp', 'cbc']
GHS_TRAIT_SHORT = ['Cab', 'Car', 'Anth', 'Cw', 'Cm', 'LAI', 'Cp', 'Cbc']
GHS_N_TRAITS = 8
GHS_DATA_DIR = Path('data/GreenHyperSpectra')
CHERIF_DATA_DIR = Path('multi-traitretrieval/dataset')

N_TRAITS = 20
N_BANDS = 1721
BAND_START = '400'
BAND_END = '2450'

# Maps dataset identifier to its default data directory
_DATASET_DATA_DIRS = {
    DATASET_CHERIF: CHERIF_DATA_DIR,
    DATASET_GHS: GHS_DATA_DIR,
    DATASET_GHS_UNLABELED: GHS_DATA_DIR,
    DATASET_PROSAIL: GHS_DATA_DIR,
}


def default_data_dir(dataset: str) -> Path:
    """Return the default data directory for a dataset identifier."""
    return _DATASET_DATA_DIRS.get(dataset, CHERIF_DATA_DIR)


@dataclass
class TrainConfig:
    data_dir: Path = CHERIF_DATA_DIR
    output_dir: Path = Path('results')

    dataset: str = DATASET_CHERIF

    transform_name: str = 'cwt'
    output_size: int = 224

    model_name: str = 'efficientnet_b0'
    pretrained: bool = False

    n_folds: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    val_split: float = 0.15
    num_workers: int = 4
    seed: int = 42

    aug_baseline_shift: float = 0.02
    aug_multiplicative: float = 0.02

    accelerator: str = 'auto'
    devices: int = 1

    @property
    def is_ghs(self) -> bool:
        return self.dataset == DATASET_GHS

    @property
    def trait_names(self) -> list[str]:
        return GHS_TRAIT_NAMES if self.is_ghs else TRAIT_NAMES

    @property
    def trait_short(self) -> list[str]:
        return GHS_TRAIT_SHORT if self.is_ghs else TRAIT_SHORT

    @property
    def n_traits(self) -> int:
        return GHS_N_TRAITS if self.is_ghs else N_TRAITS

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        # GHS uses a fixed train/test split instead of k-fold CV
        if self.is_ghs:
            self.data_dir = GHS_DATA_DIR
            self.n_folds = 1
