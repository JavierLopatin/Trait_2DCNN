import numpy as np
from scipy.ndimage import zoom
from .base import BaseTransform


class MTFTransform(BaseTransform):
    """Markov Transition Field.

    Discretizes the spectrum into quantile bins, computes the Markov
    transition matrix, then maps each pair (i, j) to the transition
    probability from the bin of value_i to the bin of value_j.

    Captures sequential transition dynamics that correlation-based
    and frequency-based transforms miss.

    Output: single-channel image (H, W).
    """

    def __init__(self, output_size: int = 224, n_bins: int = 8,
                 downsample: int = 224):
        super().__init__(output_size)
        self.n_bins = n_bins
        self.downsample = downsample

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        n = len(spectrum)
        # Downsample to avoid huge matrices
        if n > self.downsample:
            indices = np.linspace(0, n - 1, self.downsample, dtype=int)
            spectrum = spectrum[indices]
            n = self.downsample

        # Quantize into bins using quantile boundaries
        bins = np.percentile(spectrum, np.linspace(0, 100, self.n_bins + 1))
        # Ensure unique bin edges
        bins = np.unique(bins)
        if len(bins) < 3:
            # Degenerate spectrum (constant or near-constant)
            mtf = np.zeros((n, n), dtype=np.float32)
        else:
            actual_bins = len(bins) - 1
            # Assign each value to a bin
            digitized = np.digitize(spectrum, bins[1:-1], right=True)
            digitized = np.clip(digitized, 0, actual_bins - 1)

            # Build transition matrix: P[i,j] = P(bin_j | bin_i)
            trans = np.zeros((actual_bins, actual_bins), dtype=np.float64)
            for t in range(n - 1):
                trans[digitized[t], digitized[t + 1]] += 1

            # Normalize rows to probabilities
            row_sums = trans.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # avoid division by zero
            trans = trans / row_sums

            # Build MTF: mtf[i,j] = transition probability from bin(x_i) to bin(x_j)
            mtf = trans[digitized][:, digitized].astype(np.float32)

        # Resize to output_size
        if mtf.shape[0] != self.output_size:
            factor = self.output_size / mtf.shape[0]
            mtf = zoom(mtf, factor, order=1)

        # Normalize to [0, 1]
        mn, mx = mtf.min(), mtf.max()
        if mx > mn:
            mtf = (mtf - mn) / (mx - mn)

        return mtf.astype(np.float32)
