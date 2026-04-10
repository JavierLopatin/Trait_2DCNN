import numpy as np
from .base import BaseTransform


class CompositeTransform(BaseTransform):
    """Stacks multiple transforms as channels in a multi-channel image.

    Each sub-transform produces (H, W) or (H, W, C) output at the same
    output_size.  All outputs are concatenated along the channel axis.

    Example:
        reshape (1ch) + cwt (1ch) + spectrogram (3ch) → 5-channel image
    """

    def __init__(self, transforms: list[BaseTransform]):
        if not transforms:
            raise ValueError("At least one transform required")
        output_size = transforms[0].output_size
        for t in transforms:
            if t.output_size != output_size:
                raise ValueError(
                    f"All transforms must have the same output_size, "
                    f"got {t.output_size} for {t.name} vs {output_size}")
        super().__init__(output_size)
        self.transforms = transforms

    @property
    def n_channels(self) -> int:
        return sum(t.n_channels for t in self.transforms)

    @property
    def name(self) -> str:
        return '+'.join(t.name for t in self.transforms)

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        channels = []
        for t in self.transforms:
            img = t.transform(spectrum)
            # Ensure 3D: (H, W) → (H, W, 1)
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            channels.append(img)
        return np.concatenate(channels, axis=-1).astype(np.float32)

    def __repr__(self):
        parts = [f"{t.name}({t.n_channels}ch)" for t in self.transforms]
        return f"CompositeTransform([{', '.join(parts)}], total={self.n_channels}ch)"
