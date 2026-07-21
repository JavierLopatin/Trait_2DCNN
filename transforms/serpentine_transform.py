import numpy as np
from .base import BaseTransform


class SerpentineTransform(BaseTransform):
    """Boustrophedon (serpentine) 1D-to-2D reshaping.

    Identical to :class:`ReshapeTransform` but fills the 2D grid row by row in
    alternating direction: row 0 left-to-right, row 1 right-to-left, and so on.
    This removes the large band jump at every row boundary of the plain reshape
    (band 41 -> 42 sits at opposite ends there), so spectrally adjacent bands
    stay spatially adjacent across rows.

    For 1721 bands -> 42x42 = 1764 (43 zero-padded), then resized to output_size.
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
        # Reverse odd rows so the raster snakes back and forth
        img[1::2] = img[1::2, ::-1]

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
