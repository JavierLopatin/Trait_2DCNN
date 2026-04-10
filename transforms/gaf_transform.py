import numpy as np
from scipy.ndimage import zoom
from .base import BaseTransform


class GAFTransform(BaseTransform):
    """Gramian Angular Field (GASF + GADF).

    Maps signal values to angular coordinates and computes the Gram matrix.
    GASF: cos(phi_i + phi_j) — captures correlations.
    GADF: sin(phi_i - phi_j) — captures differences.

    Output: 2-channel image (GASF, GADF).
    """

    def __init__(self, output_size: int = 224, downsample: int = 224):
        super().__init__(output_size)
        self.downsample = downsample

    @property
    def n_channels(self) -> int:
        return 2

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        # Downsample spectrum first to avoid huge (1721x1721) matrices
        n = len(spectrum)
        if n > self.downsample:
            indices = np.linspace(0, n - 1, self.downsample, dtype=int)
            spectrum = spectrum[indices]

        # Normalize to [-1, 1]
        vmin, vmax = spectrum.min(), spectrum.max()
        if vmax > vmin:
            scaled = 2 * (spectrum - vmin) / (vmax - vmin) - 1
        else:
            scaled = np.zeros_like(spectrum)

        # Clip for numerical stability with arccos
        scaled = np.clip(scaled, -1, 1)
        phi = np.arccos(scaled)

        # GASF: cos(phi_i + phi_j)
        gasf = np.cos(phi[:, None] + phi[None, :])
        # GADF: sin(phi_i - phi_j)
        gadf = np.sin(phi[:, None] - phi[None, :])

        # Resize to output_size
        def resize_and_norm(mat):
            if mat.shape[0] != self.output_size:
                factor = self.output_size / mat.shape[0]
                mat = zoom(mat, factor, order=1)
            # Normalize to [0, 1]
            mn, mx = mat.min(), mat.max()
            if mx > mn:
                mat = (mat - mn) / (mx - mn)
            return mat.astype(np.float32)

        return np.stack([resize_and_norm(gasf), resize_and_norm(gadf)], axis=-1)
