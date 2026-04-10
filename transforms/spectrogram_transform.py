import numpy as np
from scipy.signal import stft
from scipy.ndimage import zoom
from .base import BaseTransform


class SpectrogramTransform(BaseTransform):
    """Multi-Channel Spectrogram via STFT.

    Applies STFT with 3 different window functions (Bartlett, Gaussian, Blackman)
    and stacks them as RGB channels. Based on Shuai et al. (2025).

    Output: 3-channel image (one per window function).
    """

    def __init__(self, output_size: int = 224, nperseg: int = 64):
        super().__init__(output_size)
        self.nperseg = nperseg
        self.windows = ['bartlett', ('gaussian', 12), 'blackman']

    @property
    def n_channels(self) -> int:
        return 3

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        channels = []
        for window in self.windows:
            _, _, Zxx = stft(spectrum, nperseg=self.nperseg, window=window,
                             noverlap=self.nperseg // 2)
            mag = np.abs(Zxx)

            # Resize to output_size x output_size
            zoom_factors = (self.output_size / mag.shape[0],
                            self.output_size / mag.shape[1])
            mag = zoom(mag, zoom_factors, order=1)

            # Normalize to [0, 1]
            vmin, vmax = mag.min(), mag.max()
            if vmax > vmin:
                mag = (mag - vmin) / (vmax - vmin)

            channels.append(mag.astype(np.float32))

        return np.stack(channels, axis=-1)
