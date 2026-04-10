import numpy as np
from scipy.ndimage import zoom
from .base import BaseTransform


class COS2DTransform(BaseTransform):
    """2D Correlation Spectroscopy (2D-COS).

    Computes synchronous and asynchronous correlation maps from a single
    spectrum using perturbation via spectral derivatives.
    Synchronous = captures in-phase inter-band correlations.
    Asynchronous = captures sequential spectral changes.

    Output: 2-channel image (sync, async) resized to output_size.
    """

    def __init__(self, output_size: int = 224, n_segments: int = 10):
        super().__init__(output_size)
        self.n_segments = n_segments

    @property
    def n_channels(self) -> int:
        return 2

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        n_bands = len(spectrum)

        # Create perturbation matrix from derivatives and segments
        # Split spectrum into overlapping segments to simulate perturbation
        seg_len = n_bands // self.n_segments
        segments = []
        for i in range(self.n_segments):
            start = i * seg_len
            end = min(start + seg_len + seg_len // 2, n_bands)
            seg = np.zeros(n_bands)
            seg[start:end] = spectrum[start:end]
            segments.append(seg)

        X = np.array(segments)  # (n_segments, n_bands)
        mean_spec = X.mean(axis=0)
        X_centered = X - mean_spec

        # Synchronous correlation
        sync = X_centered.T @ X_centered / (self.n_segments - 1)

        # Asynchronous correlation via Hilbert-Noda transform
        N = self.n_segments
        noda = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    noda[i, j] = 1.0 / (np.pi * (j - i))
        async_corr = X_centered.T @ noda @ X_centered / (N - 1)

        # Resize both to output_size
        def resize_and_norm(mat):
            zoom_factors = (self.output_size / mat.shape[0],
                            self.output_size / mat.shape[1])
            resized = zoom(mat, zoom_factors, order=1)
            vmin, vmax = resized.min(), resized.max()
            if vmax > vmin:
                resized = (resized - vmin) / (vmax - vmin)
            return resized.astype(np.float32)

        sync_img = resize_and_norm(sync)
        async_img = resize_and_norm(async_corr)

        return np.stack([sync_img, async_img], axis=-1)
