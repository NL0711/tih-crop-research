from .DAMamba import DAMamba
from .resnet import ResNet50


def build_model(config, is_pretrain=False):
    """Build and return a model instance based on config.MODEL.TYPE.

    Supported MODEL.TYPE values:
      - "DAMamba"  : Original DAMamba architecture
      - "resnet50" : ResNet-50 with ImageNet-1K V2 pretrained weights
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
        # torchvision loads ImageNet-1K V2 weights internally when
        # pretrained=True. We do NOT use config.MODEL.PRETRAINED here —
        # that field being empty prevents main.py from calling
        # load_pretrained_ema() (which is DAMamba-specific).
        model = ResNet50(
            num_classes=config.MODEL.NUM_CLASSES,
            pretrained=True,        # Always use ImageNet-1K V2 weights
            drop_rate=config.MODEL.DROP_RATE,
        )
        return model

    # ------------------------------------------------------------------
    # Unknown model type
    # ------------------------------------------------------------------
    raise ValueError(
        f"Unknown MODEL.TYPE '{model_type}'. "
        f"Supported types: DAMamba, resnet50"
    )
