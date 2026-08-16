from .DAMamba import DAMamba
from .resnet import ResNet50
from .vmamba import VMamba, VMamba_T
from .inception import InceptionV3, inceptionv3
from .efficientnet import EfficientNetB1, efficientnet_b1_model
from .inception_resnet_v2 import InceptionResNetV2, inception_resnet_v2_model


def build_model(config, is_pretrain=False):
    """Build and return a model instance based on config.MODEL.TYPE.

    Supported MODEL.TYPE values:
      - "DAMamba"                  : Original DAMamba architecture
      - "resnet50"                 : ResNet-50 with ImageNet-1K V2 pretrained weights
      - "inception_v3"             : Inception-v3 with ImageNet-1K pretrained weights
      - "efficientnet_b1"          : EfficientNet-B1 with ImageNet-1K V2 pretrained weights
      - "inception_resnet_v2"      : Inception-ResNet-v2 with ImageNet-1K pretrained weights
      - "vmamba_t" / "vmamba_tiny" : VMamba-T (Visual State Space Model - Tiny)
    """
    model_type = config.MODEL.TYPE

    # ------------------------------------------------------------------
    # DAMamba family
    # ------------------------------------------------------------------
    if model_type in ["DAMamba"]:
        model = DAMamba(
            in_chans=config.MODEL.DAMAMBA.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.DAMAMBA.DEPTHS,
            dims=config.MODEL.DAMAMBA.EMBED_DIM,
            mlp_ratios=config.MODEL.DAMAMBA.MLP_RATIO,
            head_dim=config.MODEL.DAMAMBA.HEAD_DIM,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layerscale=config.MODEL.DAMAMBA.LAYERSCALE,
            hybrid_enable=config.MODEL.HYBRID.ENABLE
        )
        return model

    # ------------------------------------------------------------------
    # CNN baselines
    # ------------------------------------------------------------------
    elif model_type == "resnet50":
        model = ResNet50(
            num_classes=config.MODEL.NUM_CLASSES,
            pretrained=True,        # Always use ImageNet-1K V2 weights
            drop_rate=config.MODEL.DROP_RATE,
        )
        return model

    elif model_type in ["inception_v3", "inceptionv3", "inception"]:
        freeze_backbone = getattr(config.MODEL, "FREEZE_BACKBONE", False)
        freeze_stages = getattr(config.MODEL, "FREEZE_STAGES", -1)
        model = InceptionV3(
            num_classes=config.MODEL.NUM_CLASSES,
            pretrained=True,        # Always use ImageNet-1K weights
            drop_rate=config.MODEL.DROP_RATE,
            freeze_backbone=freeze_backbone,
            freeze_stages=freeze_stages,
        )
        return model

    elif model_type in ["efficientnet_b1", "efficientnetb1", "efficientnet"]:
        freeze_backbone = getattr(config.MODEL, "FREEZE_BACKBONE", False)
        freeze_stages = getattr(config.MODEL, "FREEZE_STAGES", -1)
        model = EfficientNetB1(
            num_classes=config.MODEL.NUM_CLASSES,
            pretrained=True,        # Always use ImageNet-1K V2 weights
            drop_rate=config.MODEL.DROP_RATE,
            freeze_backbone=freeze_backbone,
            freeze_stages=freeze_stages,
        )
        return model

    elif model_type in ["inception_resnet_v2", "inceptionresnetv2", "inception_resnet"]:
        freeze_backbone = getattr(config.MODEL, "FREEZE_BACKBONE", False)
        freeze_stages = getattr(config.MODEL, "FREEZE_STAGES", -1)
        model = InceptionResNetV2(
            num_classes=config.MODEL.NUM_CLASSES,
            pretrained=True,        # Always use ImageNet-1K weights
            drop_rate=config.MODEL.DROP_RATE,
            freeze_backbone=freeze_backbone,
            freeze_stages=freeze_stages,
        )
        return model

    # ------------------------------------------------------------------
    # VMamba family
    # ------------------------------------------------------------------
    elif model_type in ["vmamba_t", "vmamba_tiny", "vmamba"]:
        pretrained_path = config.MODEL.PRETRAINED if (hasattr(config.MODEL, "PRETRAINED") and config.MODEL.PRETRAINED) else None
        freeze_backbone = getattr(config.MODEL, "FREEZE_BACKBONE", False)
        freeze_stages = getattr(config.MODEL, "FREEZE_STAGES", -1)
        model = VMamba_T(
            num_classes=config.MODEL.NUM_CLASSES,
            pretrained=True,        # Use official ImageNet-1K pretrained weights
            pretrained_path=pretrained_path,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            freeze_backbone=freeze_backbone,
            freeze_stages=freeze_stages,
        )
        return model

    # ------------------------------------------------------------------
    # Unknown model type
    # ------------------------------------------------------------------
    raise ValueError(
        f"Unknown MODEL.TYPE '{model_type}'. "
        f"Supported types: DAMamba, resnet50, inception_v3, efficientnet_b1, inception_resnet_v2, vmamba_t"
    )
