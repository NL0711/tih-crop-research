## 1. Class-Count Fix

- [x] 1.1 In `data/build.py`, derive `nb_classes = len(dataset.classes)` for the ImageFolder (imagenet) path in `build_dataset()` and remove the hardcoded `nb_classes = 1000`, so `build_loader()` stops overwriting `config.MODEL.NUM_CLASSES` with 1000.

## 2. Hybrid Wiring Fix

- [x] 2.1 In `models/__init__.py`, forward the existing flag by adding `hybrid_enable=config.MODEL.HYBRID.ENABLE` to the `DAMamba(...)` call in `build_model()`.

## 3. Safe Pretrained-Load Fix

- [x] 3.1 In `utils/utils.py`, pre-filter the checkpoint state dict inside `load_pretrained_ema` to drop keys whose tensor shape is incompatible with the model (only `head.fc.weight`/`head.fc.bias` for a 1000→5 mismatch), then load with `strict=False`. Preserve name-level validation; apply the same filter to the EMA load branch.

## 4. Static Verification

- [x] 4.1 Confirm `len(dataset.classes) == 5` for the cauliflower dataset (folder layout) and that `build_loader()` sets `config.MODEL.NUM_CLASSES` to the derived count instead of 1000.
- [x] 4.2 Confirm `build_model()` consumes `MODEL.NUM_CLASSES == 5` and that `DAMamba.head.fc` has output dimension 5.
- [x] 4.3 Confirm `MODEL.HYBRID.ENABLE=True` reaches `DAMamba(..., hybrid_enable=True)` and `False` still constructs the vanilla model (no `local_cnn`/`fusion`).
- [x] 4.4 Confirm the 1000→5 classifier mismatch loads safely: `strict=False` alone raises on torch 2.2+ (size mismatch), so the `load_pretrained_ema` filter must drop the incompatible head keys, load the backbone, and leave `head.fc` at its initialization — without weakening name-level checkpoint validation.