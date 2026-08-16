import numpy as np
from .base import BaseTransform


class ReshapeTransform(BaseTransform):
    """Direct 1D-to-2D reshaping baseline.

    Simply reshapes the spectral vector into a square-ish 2D matrix.
    For 1721 bands → 42x42 = 1764 (43 zero-padded).
    Then resizes to output_size if needed.

    normalize='local' (default) scales each image to [0, 1] using its own
    min/max. normalize='global' uses a fixed min/max computed once over the
    training spectra via fit() instead.
    """

    def __init__(self, output_size: int = 224, normalize: str = 'local'):
        super().__init__(output_size)
        if normalize not in ('local', 'global'):
            raise ValueError(f"normalize must be 'local' or 'global', got {normalize!r}")
        self.normalize = normalize
        self.global_min = None
        self.global_max = None

    def fit(self, spectra: np.ndarray):
        """Compute dataset-wide min/max, used when normalize='global'."""
        self.global_min = float(np.min(spectra))
        self.global_max = float(np.max(spectra))

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
        if self.normalize == 'global':
            if self.global_min is None or self.global_max is None:
                raise RuntimeError(
                    "normalize='global' requires fit(spectra) before transform()")
            vmin, vmax = self.global_min, self.global_max
        else:
            vmin, vmax = img.min(), img.max()

        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)

        return img.astype(np.float32)


class ReshapeGlobalTransform(ReshapeTransform):
    """ReshapeTransform preset with normalize='global'."""

    def __init__(self, output_size: int = 224):
        super().__init__(output_size, normalize='global')
