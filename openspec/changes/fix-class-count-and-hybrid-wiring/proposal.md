## Why

A fine-tuned DAMamba-T on a 5-class cauliflower disease dataset immediately produces Loss ≈ 8.34 and 0% Top-1/Top-5 after loading the ImageNet-1k pretrained checkpoint — before any training. Two wiring bugs make the model functionally wrong: the class count is forced to 1000 (ImageNet) instead of the dataset's real 5 classes, and the CNN hybrid branch never activates because the config flag is never forwarded to the model constructor.

## What Changes

- **data/build.py**: For `ImageFolder`-based datasets, derive `nb_classes` from the actual dataset (`len(dataset.classes)`) instead of the hardcoded `nb_classes = 1000`. This stops `build_loader()` from silently overwriting `config.MODEL.NUM_CLASSES`.
- **models/__init__.py**: Forward the existing `config.MODEL.HYBRID.ENABLE` flag into the `DAMamba` constructor as `hybrid_enable=...` so the local-CNN/fusion branch is instantiated and exercised when enabled.
- No changes to LR, optimizer, scheduler, augmentation, epochs, checkpoints, pretrained weights, or dataset contents.
- Backward compatible: when `MODEL.HYBRID.ENABLE=False`, behavior is byte-for-byte identical to the current vanilla path.

## Capabilities

### New Capabilities

- `class-count-mapping`: Derives the model classifier dimension from the actual ImageFolder dataset classes so `MODEL.NUM_CLASSES` reflects the real label space for train/val/test.
- `hybrid-model-wiring`: Propagates the existing `MODEL.HYBRID.ENABLE` config through `build_model` to instantiate and run the local-CNN + fusion hybrid branch when enabled.

### Modified Capabilities

<!-- None: no main specs exist yet. -->

## Impact

- `classification/data/build.py` — `build_dataset()` return value for the `imagenet` ImageFolder path.
- `classification/models/__init__.py` — `build_model()` kwargs.
- Downstream effects: `DAMamba.head` output dimension becomes the dataset class count (5 for cauliflower); the ImageNet 1000-way `head.fc` is skipped by the existing `strict=False` loading path; hybrid runs add `local_cnn`/`fusion` weights (untrained at load) which the pretrained checkpoint will not supply.