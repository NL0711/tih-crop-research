"""
inception.py
============
Inception-v3 CNN Baseline Architecture with ImageNet-1K Pretrained Weights.

Paper: "Rethinking the Inception Architecture for Computer Vision"
       Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z. (CVPR 2016)
       https://arxiv.org/abs/1512.00567

Pretrained Weights:
  - torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
  - ImageNet-1K Top-1 Accuracy: 77.29%, Top-5: 93.45%
  - Backbone: Multi-branch Inception modules with factorized 1x7, 7x1 convolutions and grid size reductions.
  - Adapted classifier: Linear(2048, num_classes=5)
"""

import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights


def freeze_inception(model: nn.Module, freeze_backbone: bool = False, freeze_stages: int = -1):
    """Freeze backbone weights or modules for fast downstream fine-tuning."""
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        print("[Fine-Tuning] Frozen entire Inception-v3 backbone. Only classifier head (fc) will be updated.")
    elif freeze_stages >= 0:
        # Stages in Inception-v3:
        # Stage 0: Stem (Conv2d_1a_3x3 through Conv2d_4a_3x3, maxpool1, maxpool2)
        # Stage 1: Mixed_5b, Mixed_5c, Mixed_5d (Inception A blocks)
        # Stage 2: Mixed_6a through Mixed_6e (Inception B blocks)
        # Stage 3: Mixed_7a through Mixed_7c (Inception C blocks)
        stem_modules = [
            model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
            model.maxpool1, model.Conv2d_3b_1x1, model.Conv2d_4a_3x3, model.maxpool2
        ]
        if freeze_stages >= 0:
            for m in stem_modules:
                m.requires_grad_(False)
        if freeze_stages >= 1:
            for m in [model.Mixed_5b, model.Mixed_5c, model.Mixed_5d]:
                m.requires_grad_(False)
        if freeze_stages >= 2:
            for m in [model.Mixed_6a, model.Mixed_6b, model.Mixed_6c, model.Mixed_6d, model.Mixed_6e]:
                m.requires_grad_(False)
        print(f"[Fine-Tuning] Frozen Inception-v3 modules through stage {freeze_stages}.")


class InceptionV3(nn.Module):
    """Inception-v3 wrapper supporting both 224x224 and 299x299 inputs,

    ImageNet-1K pretrained weights, and seamless integration into standard
    classification pipelines.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        freeze_backbone: bool = False,
        freeze_stages: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            weights = Inception_V3_Weights.IMAGENET1K_V1
            self.model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        else:
            self.model = inception_v3(weights=None, aux_logits=True, transform_input=False)

        # Disable AuxLogits to ensure deterministic single-tensor forward output
        self.model.aux_logits = False
        self.model.AuxLogits = None

        # Replace classification head
        in_features = self.model.fc.in_features  # 2048
        if drop_rate > 0.0:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=drop_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone or freeze_stages >= 0:
            freeze_inception(self.model, freeze_backbone=freeze_backbone, freeze_stages=freeze_stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # Handle tuple output if aux_logits was somehow triggered
        if isinstance(out, tuple):
            return out[0]
        return out


def inceptionv3(
    num_classes: int = 5,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    freeze_backbone: bool = False,
    freeze_stages: int = -1,
    **kwargs,
) -> InceptionV3:
    """Factory function for Inception-v3 with ImageNet-1K weights."""
    return InceptionV3(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        freeze_backbone=freeze_backbone,
        freeze_stages=freeze_stages,
        **kwargs,
    )
