import timm
import torch
import torch.nn as nn
from typing import Optional


def create_model(model_name: str, n_traits: int = 20, in_channels: int = 1,
                 pretrained: bool = False, drop_rate: float = 0.3) -> nn.Module:
    """Create a 2D model for multi-trait regression using timm.

    Args:
        model_name: timm model name (e.g., 'efficientnet_b0', 'resnet50',
                    'convnext_tiny', 'vit_small_patch16_224', 'swin_tiny_patch4_window7_224')
        n_traits: Number of output traits
        in_channels: Number of input channels (1=grayscale, 2=sync/async, 3=RGB)
        pretrained: Use ImageNet pretrained weights (only works with 3 channels)
        drop_rate: Dropout rate before final layer
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=n_traits,
        drop_rate=drop_rate,
    )
    return model


# Model registry with default configs
MODEL_REGISTRY = {
    # CNN models
    'efficientnet_b0': {'model_name': 'efficientnet_b0'},
    'efficientnet_v2_s': {'model_name': 'tf_efficientnetv2_s'},
    'resnet50': {'model_name': 'resnet50'},
    'convnext_tiny': {'model_name': 'convnext_tiny'},
    # Transformer models
    'vit_small': {'model_name': 'vit_small_patch16_224'},
    'swin_tiny': {'model_name': 'swin_tiny_patch4_window7_224'},
    # Hybrid
    'coatnet_0': {'model_name': 'coatnet_0_rw_224'},
}


def get_model(name: str, in_channels: int = 1, n_traits: int = 20,
              pretrained: bool = False) -> nn.Module:
    """Get model by short name from registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    cfg = MODEL_REGISTRY[name]
    return create_model(cfg['model_name'], n_traits=n_traits,
                        in_channels=in_channels, pretrained=pretrained)
