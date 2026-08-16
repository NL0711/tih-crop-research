"""
inception_resnet_v2.py
======================
Inception-ResNet-v2 CNN Baseline Architecture with ImageNet-1K Pretrained Weights.

Paper: "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"
       Szegedy, C., Ioffe, S., Vanhoucke, V., Alemi, A. A. (AAAI 2017)
       https://arxiv.org/abs/1602.07261

Pretrained Weights:
  - timm.create_model('inception_resnet_v2', pretrained=True)
  - ImageNet-1K Top-1 Accuracy: 80.40%, Top-5: 95.30%
  - Architecture: Residual Inception blocks (Inception-ResNet-A, B, C) with residual scaling.
  - Native spatial resolution: 299x299 (also supports 224x224).
  - Adapted classifier: Linear(1536, num_classes=5)
"""

import timm
import torch
import torch.nn as nn


def freeze_inception_resnet_v2(model: nn.Module, freeze_backbone: bool = False, freeze_stages: int = -1):
    """Freeze backbone weights or modules for fast downstream fine-tuning."""
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not ("classif" in name or "head" in name or "fc" in name):
                param.requires_grad = False
        print("[Fine-Tuning] Frozen entire Inception-ResNet-v2 backbone. Only classifier head will be updated.")
    elif freeze_stages >= 0:
        # Stages in Inception-ResNet-v2 (timm):
        # Stage 0: conv2d_1a..conv2d_4b, maxpool_3a, maxpool_5a (stem)
        # Stage 1: repeat (Inception-ResNet-A blocks)
        # Stage 2: mixed_6a, repeat_1 (Inception-ResNet-B blocks)
        # Stage 3: mixed_7a, repeat_2, block8, conv2d_7b (Inception-ResNet-C blocks + reduction)
        stem_attrs = [
            "conv2d_1a", "conv2d_2a", "conv2d_2b", "maxpool_3a",
            "conv2d_3b", "conv2d_4a", "maxpool_5a"
        ]
        if freeze_stages >= 0:
            for attr in stem_attrs:
                if hasattr(model, attr):
                    getattr(model, attr).requires_grad_(False)
        if freeze_stages >= 1 and hasattr(model, "repeat"):
            model.repeat.requires_grad_(False)
        if freeze_stages >= 2:
            if hasattr(model, "mixed_6a"):
                model.mixed_6a.requires_grad_(False)
            if hasattr(model, "repeat_1"):
                model.repeat_1.requires_grad_(False)
        print(f"[Fine-Tuning] Frozen Inception-ResNet-v2 through stage {freeze_stages}.")


class InceptionResNetV2(nn.Module):
    """Inception-ResNet-v2 wrapper supporting ImageNet-1K pretrained weights,

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

        self.model = timm.create_model(
            "inception_resnet_v2",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate if drop_rate > 0.0 else 0.0,
        )

        if freeze_backbone or freeze_stages >= 0:
            freeze_inception_resnet_v2(
                self.model, freeze_backbone=freeze_backbone, freeze_stages=freeze_stages
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def inception_resnet_v2_model(
    num_classes: int = 5,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    freeze_backbone: bool = False,
    freeze_stages: int = -1,
    **kwargs,
) -> InceptionResNetV2:
    """Factory function for Inception-ResNet-v2 with ImageNet-1K weights."""
    return InceptionResNetV2(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        freeze_backbone=freeze_backbone,
        freeze_stages=freeze_stages,
        **kwargs,
    )
