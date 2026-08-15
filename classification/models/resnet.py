"""
resnet.py
=========
ResNet-50 CNN baseline for the DAMamba crop-disease classification benchmark.

Wraps torchvision.models.resnet50 with:
  - ImageNet-1K V2 pretrained weights (80.86% top-1 on ImageNet)
  - Custom classification head adapted to the target num_classes
  - Optional dropout before the final linear layer
  - No flops() method (main.py guards with hasattr(model, 'flops'))
  - No no_weight_decay() method (optimizer applies standard weight decay)

References
----------
- Paper:   "Deep Residual Learning for Image Recognition" (He et al., 2016)
           https://arxiv.org/abs/1512.03385
- Weights: torchvision ResNet50_Weights.IMAGENET1K_V2
           https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
"""

import torch
import torch.nn as nn


class ResNet50(nn.Module):
    """ResNet-50 with a custom classification head.

    Args:
        num_classes (int): Number of output classes. Default: 5.
        pretrained (bool): If True, load ImageNet-1K V2 weights from torchvision.
                           The original 1000-class fc layer is replaced before
                           any weights are loaded, so shape mismatches never occur.
        drop_rate (float): Dropout probability before the final linear layer.
                           Set to 0.0 to disable. Default: 0.0.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True, drop_rate: float = 0.0):
        super().__init__()

        # ------------------------------------------------------------------
        # Load backbone (with or without ImageNet weights)
        # ------------------------------------------------------------------
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
        except (ImportError, AttributeError):
            # torchvision < 0.13 fallback
            from torchvision.models import resnet50 as _resnet50
            backbone = _resnet50(pretrained=pretrained)

        # ------------------------------------------------------------------
        # Replace the classifier head
        # ------------------------------------------------------------------
        in_features = backbone.fc.in_features  # 2048 for ResNet-50

        if drop_rate > 0.0:
            backbone.fc = nn.Sequential(
                nn.Dropout(p=drop_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            backbone.fc = nn.Linear(in_features, num_classes)

        self.model = backbone
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}"
