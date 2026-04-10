import numpy as np
import pywt
from scipy.ndimage import zoom
from .base import BaseTransform


class CWTTransform(BaseTransform):
    """Continuous Wavelet Transform scalogram.

    Convolves the spectrum with wavelets at multiple scales to produce
    a 2D time-scale representation (scalogram).
    """

    def __init__(self, output_size: int = 224, wavelet: str = 'morl',
                 n_scales: int = 128):
        super().__init__(output_size)
        self.wavelet = wavelet
        self.n_scales = n_scales
        self.scales = np.arange(1, n_scales + 1)

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        coeffs, _ = pywt.cwt(spectrum, self.scales, self.wavelet)
        # coeffs shape: (n_scales, n_bands)
        scalogram = np.abs(coeffs)

        # Resize to output_size x output_size
        zoom_factors = (self.output_size / scalogram.shape[0],
                        self.output_size / scalogram.shape[1])
        scalogram = zoom(scalogram, zoom_factors, order=1)

        # Normalize to [0, 1]
        vmin, vmax = scalogram.min(), scalogram.max()
        if vmax > vmin:
            scalogram = (scalogram - vmin) / (vmax - vmin)

        return scalogram.astype(np.float32)
