from .base import BaseTransform
from .cwt_transform import CWTTransform
from .cos2d_transform import COS2DTransform
from .reshape_transform import ReshapeTransform, ReshapeGlobalTransform
from .serpentine_transform import SerpentineTransform
from .hilbert_transform import HilbertTransform
from .ndi_transform import NDITransform
from .gaf_transform import GAFTransform
from .spectrogram_transform import SpectrogramTransform
from .mtf_transform import MTFTransform
from .composite_transform import CompositeTransform

__all__ = [
    'BaseTransform', 'CompositeTransform', 'CWTTransform', 'COS2DTransform',
    'ReshapeTransform', 'ReshapeGlobalTransform', 'SerpentineTransform', 'HilbertTransform',
    'NDITransform', 'GAFTransform', 'SpectrogramTransform', 'MTFTransform', 'TRANSFORMS',
]

TRANSFORMS = {
    'cwt': CWTTransform,
    'cos2d': COS2DTransform,
    'reshape': ReshapeTransform,
    'reshape_global': ReshapeGlobalTransform,
    'serpentine': SerpentineTransform,
    'hilbert': HilbertTransform,
    'ndi': NDITransform,
    'gaf': GAFTransform,
    'spectrogram': SpectrogramTransform,
    'mtf': MTFTransform,
}
