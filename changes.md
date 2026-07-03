# Changes since init commit

## Added
- `.gitignore` to ignore Python artifacts, build outputs, model checkpoints, logs, and temporary files.
- `changes.md` to summarize codebase changes.
- `classification/utils/split_dataset.py` to split a dataset by class folders into `train`, `val`, and `test` sets.
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

## Notes
- Existing shell scripts (`*.sh`) remain available for Linux/macOS use.
- PowerShell scripts provide a Windows-friendly alternative for distributed training/testing and custom op build steps.
- The new split utility supports dataset roots where class subfolders correspond to disease categories.
- `triton` is skipped during dependency install on Windows.

