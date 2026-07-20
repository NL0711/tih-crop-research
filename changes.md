# Changes since init commit

## Added
- `.gitignore` to ignore Python artifacts, build outputs, model checkpoints, logs, and temporary files.
- `changes.md` to summarize codebase changes.
- `classification/utils/split_dataset.py` to split a dataset by class folders into `train`, `val`, and `test` sets.
- class imbalance support in `classification/data/build.py` via class-balanced oversampling for the training dataset, exposed through the new `--oversample` CLI flag and `DATA.OVERSAMPLE` config option.
- PowerShell wrappers for distributed training/test scripts:
  - `detection/maskrcnn/dist_train.ps1`
  - `detection/maskrcnn/dist_test.ps1`
  - `segmentation/upernet/dist_train.ps1`
  - `segmentation/upernet/dist_test.ps1`
- PowerShell build wrappers for custom ops:
  - `classification/models/ops_dcnv3/make.ps1`
  - `detection/maskrcnn/ops_dcnv3/make.ps1`
  - `segmentation/upernet/ops_dcnv3/make.ps1`
- `classification/pyproject.toml` to define the classification package metadata.

## Modified
- `classification/config.py`
  - changed `MODEL.NUM_CLASSES` to `5`
  - increased `MODEL.DROP_RATE` to `0.2`
  - increased `MODEL.DROP_PATH_RATE` to `0.3`
  - increased `MODEL.LABEL_SMOOTHING` to `0.2`
  - increased `AUG.REPROB` to `0.5`
- `classification/configs/DAMamba/damamba_base.yaml`
  - reduced `TRAIN.EPOCHS` from `300` to `50`
  - reduced `TRAIN.WARMUP_EPOCHS` from `20` to `3`
- `classification/configs/DAMamba/damamba_small.yaml`
  - reduced `TRAIN.EPOCHS` from `300` to `50`
  - reduced `TRAIN.WARMUP_EPOCHS` from `20` to `3`
- `classification/requirements.txt`
  - changed `triton` to `triton; sys_platform != "win32"` for Windows compatibility.

## Added
- `attention.py` in `classification/models/`, `detection/maskrcnn/`, and `segmentation/upernet/` implementing:
  - `SEBlock` (Squeeze-and-Excitation attention)
  - `CBAM` (Convolutional Block Attention Module)
  - `StageAttentionWrapper` to apply attention to multi-stage features with support for SE ablation study.
- `classification/data/enhancement.py` implementing:
  - `AdvancedImageEnhancer`: A data preprocessing layer utilizing OpenCV and Albumentations. It performs basic enhancements (brightness/contrast, CLAHE histogram equalization, edge sharpening, bilateral noise reduction, and HSV color balance/saturation correction) as well as advanced noise and blur injection (Gaussian/median/motion blur, salt-and-pepper, and Poisson noise).

## Modified
- `DAMamba.py` in `classification/models/`, `detection/maskrcnn/`, and `segmentation/upernet/`:
  - Added `use_attention` and `use_se_only` flag support in the initialization to easily integrate and toggle attention modules.
  - Hooked `StageAttentionWrapper` in the forward pass (`forward_features`) of the backbone to refine intermediate stage representations before head/downstream layers.
- `classification/config.py`:
  - Added new `DATA.ENHANCEMENT` configuration parameters for controlling basic and advanced enhancement features.
- `classification/data/build.py` and `classification/data/data_simmim_ft.py`:
  - Integrated `AdvancedImageEnhancer` as the first step of the dataset transformation pipeline for both training and validation tasks when enabled.

## Notes
- Existing shell scripts (`*.sh`) remain available for Linux/macOS use.
- PowerShell scripts provide a Windows-friendly alternative for distributed training/testing and custom op build steps.
- The new split utility supports dataset roots where class subfolders correspond to disease categories.
- `triton` is skipped during dependency install on Windows.
- Attention modules are fully convolutional and can process feature maps with arbitrary spatial resolutions.
- Preprocessing and enhancement layers allow adjusting parameters dynamically using config files.

