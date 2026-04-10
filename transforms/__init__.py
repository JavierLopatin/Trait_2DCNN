from .base import BaseTransform
from .cwt_transform import CWTTransform
from .cos2d_transform import COS2DTransform
from .reshape_transform import ReshapeTransform
from .gaf_transform import GAFTransform
from .spectrogram_transform import SpectrogramTransform
from .mtf_transform import MTFTransform
from .composite_transform import CompositeTransform

TRANSFORMS = {
    'cwt': CWTTransform,
    'cos2d': COS2DTransform,
    'reshape': ReshapeTransform,
    'gaf': GAFTransform,
    'spectrogram': SpectrogramTransform,
    'mtf': MTFTransform,
}
