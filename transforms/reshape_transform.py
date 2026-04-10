import numpy as np
from .base import BaseTransform


class ReshapeTransform(BaseTransform):
    """Direct 1D-to-2D reshaping baseline.

    Simply reshapes the spectral vector into a square-ish 2D matrix.
    For 1721 bands → 42x42 = 1764 (43 zero-padded).
    Then resizes to output_size if needed.
    """

    def __init__(self, output_size: int = 224):
        super().__init__(output_size)

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        n = len(spectrum)
        side = int(np.ceil(np.sqrt(n)))

        # Zero-pad to fill the square
        padded = np.zeros(side * side, dtype=np.float32)
        padded[:n] = spectrum

        img = padded.reshape(side, side)

        # Resize to output_size if different
        if side != self.output_size:
            from scipy.ndimage import zoom
            factor = self.output_size / side
            img = zoom(img, factor, order=1)

        # Normalize to [0, 1]
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)

        return img.astype(np.float32)
