import numpy as np
from scipy.ndimage import zoom
from .base import BaseTransform


class NDITransform(BaseTransform):
    """All-band normalized-difference matrix (3 channels).

    Builds an exhaustive two-band spectral-index map: every pixel (i, j)
    encodes a relationship between bands lambda_i and lambda_j. This mimics the
    principle of vegetation indices (e.g. NDVI), where trait-sensitive band
    pairs are found via all-pairs normalized-difference maps, and directly
    tests the long-range inter-band interaction hypothesis.

    Because the normalized difference removes absolute reflectance information,
    three complementary channels are stacked:
      1. (R_i - R_j) / (R_i + R_j + eps)  -- normalized difference (index)
      2. |R_i - R_j|                      -- absolute difference (magnitude)
      3. sqrt(R_i * R_j)                  -- geometric mean (absolute level)

    The 1721-band spectrum is downsampled to ``downsample`` bands first to avoid
    a 1721x1721 matrix. Output: (output_size, output_size, 3).
    """

    def __init__(self, output_size: int = 224, downsample: int = 224,
                 eps: float = 1e-8):
        super().__init__(output_size)
        self.downsample = downsample
        self.eps = eps

    @property
    def n_channels(self) -> int:
        return 3

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        # Downsample the spectrum first to keep the matrix tractable.
        n = len(spectrum)
        if n > self.downsample:
            indices = np.linspace(0, n - 1, self.downsample, dtype=int)
            s = spectrum[indices].astype(np.float64)
        else:
            s = spectrum.astype(np.float64)

        ri = s[:, None]
        rj = s[None, :]

        ndi = (ri - rj) / (ri + rj + self.eps)   # normalized difference
        adiff = np.abs(ri - rj)                   # absolute magnitude
        gmean = np.sqrt(np.clip(ri * rj, 0, None))  # absolute level

        def resize_and_norm(mat):
            if mat.shape[0] != self.output_size:
                factor = self.output_size / mat.shape[0]
                mat = zoom(mat, factor, order=1)
            mn, mx = mat.min(), mat.max()
            if mx > mn:
                mat = (mat - mn) / (mx - mn)
            return mat.astype(np.float32)

        return np.stack([resize_and_norm(ndi),
                         resize_and_norm(adiff),
                         resize_and_norm(gmean)], axis=-1)
