from abc import ABC, abstractmethod
import numpy as np


class BaseTransform(ABC):
    """Base class for 1D→2D spectral transformations."""

    def __init__(self, output_size: int = 224):
        self.output_size = output_size

    @abstractmethod
    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        """Transform a single 1D spectrum into a 2D image.

        Args:
            spectrum: 1D array of shape (n_bands,)

        Returns:
            2D array of shape (H, W) or (H, W, C)
        """
        pass

    def transform_batch(self, spectra: np.ndarray) -> np.ndarray:
        """Transform a batch of spectra.

        Args:
            spectra: 2D array of shape (n_samples, n_bands)

        Returns:
            Array of shape (n_samples, H, W) or (n_samples, H, W, C)
        """
        return np.stack([self.transform(s) for s in spectra])

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace('Transform', '').lower()

    @property
    def n_channels(self) -> int:
        """Number of output channels (1 for grayscale, 3 for RGB)."""
        return 1

    def __repr__(self):
        return f"{self.__class__.__name__}(output_size={self.output_size})"
