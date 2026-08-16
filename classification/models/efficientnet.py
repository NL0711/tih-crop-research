"""
efficientnet.py
===============
EfficientNet-B1 CNN Baseline Architecture with ImageNet-1K Pretrained Weights.

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
       Tan, M., Le, Q. V. (ICML 2019)
       https://arxiv.org/abs/1905.11946

Pretrained Weights:
  - torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
  - ImageNet-1K Top-1 Accuracy: 79.84%, Top-5: 94.93%
  - Architecture: Compound scaled Mobile Inverted Bottleneck Convolution (MBConv) with Squeeze-and-Excitation.
  - Native spatial resolution: 240x240 (also supports 224x224).
  - Adapted classifier: Linear(1280, num_classes=5)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights


def freeze_efficientnet(model: nn.Module, freeze_backbone: bool = False, freeze_stages: int = -1):
    """Freeze backbone weights or stages for fast downstream fine-tuning."""
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier."):
                param.requires_grad = False
        print("[Fine-Tuning] Frozen entire EfficientNet-B1 backbone. Only classifier head will be updated.")
    elif freeze_stages >= 0:
        # Features has 9 stages: 0 (stem), 1-7 (MBConv blocks), 8 (conv head)
        for i in range(min(freeze_stages + 1, len(model.features))):
            model.features[i].requires_grad_(False)
        print(f"[Fine-Tuning] Frozen EfficientNet-B1 features through stage {freeze_stages}.")


class EfficientNetB1(nn.Module):
    """EfficientNet-B1 wrapper supporting ImageNet-1K V2 pretrained weights,

    flexible dropout, classifier adaptation, and fine-tuning freeze options.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        freeze_backbone: bool = False,
        freeze_stages: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            weights = EfficientNet_B1_Weights.IMAGENET1K_V2
            self.model = efficientnet_b1(weights=weights)
        else:
            self.model = efficientnet_b1(weights=None)

        # Replace classifier head
        in_features = self.model.classifier[1].in_features  # 1280
        dropout_p = drop_rate if drop_rate > 0.0 else 0.2
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone or freeze_stages >= 0:
            freeze_efficientnet(self.model, freeze_backbone=freeze_backbone, freeze_stages=freeze_stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def efficientnet_b1_model(
    num_classes: int = 5,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    freeze_backbone: bool = False,
    freeze_stages: int = -1,
    **kwargs,
) -> EfficientNetB1:
    """Factory function for EfficientNet-B1 with ImageNet-1K V2 weights."""
    return EfficientNetB1(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        freeze_backbone=freeze_backbone,
        freeze_stages=freeze_stages,
        **kwargs,
    )
