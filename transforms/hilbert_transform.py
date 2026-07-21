import numpy as np
from .base import BaseTransform


def _d2xy(order: int, d: int) -> tuple[int, int]:
    """Map a 1D Hilbert-curve distance ``d`` to 2D coordinates ``(x, y)``.

    Standard iterative algorithm for a ``2**order`` x ``2**order`` grid
    (Wikipedia "Hilbert curve"). Bijective over ``d in [0, 4**order)``.
    """
    x = y = 0
    s = 1
    t = d
    while s < (1 << order):
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        # Rotate the quadrant so the curve stays connected
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s <<= 1
    return x, y


class HilbertTransform(BaseTransform):
    """Hilbert space-filling-curve 1D-to-2D reshaping.

    Instead of the plain row-major raster (which breaks band locality at every
    row boundary), bands are laid out along a Hilbert curve so that spectrally
    adjacent bands map to spatially adjacent pixels throughout the whole image.

    The Hilbert curve requires a power-of-two side. We use ``side = 64``
    (4096 cells) and resample the 1721-band spectrum up to 4096 points with
    linear interpolation, which fills the grid completely (no large zero-padded
    region) while preserving the monotone band order.
    """

    def __init__(self, output_size: int = 224, order: int = 6):
        super().__init__(output_size)
        self.order = order
        self.side = 1 << order            # 2**order
        n_cells = self.side * self.side
        # Precompute the band-index -> (x, y) mapping once.
        xy = np.array([_d2xy(order, d) for d in range(n_cells)])
        self._x = xy[:, 0]
        self._y = xy[:, 1]

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        n = len(spectrum)
        n_cells = self.side * self.side

        # Resample the spectrum to exactly fill the Hilbert grid.
        if n != n_cells:
            xp = np.linspace(0, 1, n)
            xnew = np.linspace(0, 1, n_cells)
            values = np.interp(xnew, xp, spectrum).astype(np.float32)
        else:
            values = spectrum.astype(np.float32)

        img = np.zeros((self.side, self.side), dtype=np.float32)
        img[self._y, self._x] = values

        # Resize to output_size if different
        if self.side != self.output_size:
            from scipy.ndimage import zoom
            factor = self.output_size / self.side
            img = zoom(img, factor, order=1)

        # Normalize to [0, 1]
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)

        return img.astype(np.float32)
