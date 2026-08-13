## Why

Training the DAMamba classification model for the full `TRAIN.EPOCHS` budget (50 epochs in the base/small configs) wastes GPU time once validation accuracy has plateaued. There is currently no mechanism to stop training early, so runs continue to completion even when no improvement has been observed for many epochs.

## What Changes

- Add early stopping to the training loop in `classification/main.py`, monitored on validation top-1 accuracy (`acc1`).
- Stop training when validation `acc1` shows no improvement for `PATIENCE` consecutive epochs (patience = 10).
- Reset the patience counter whenever a new best validation `acc1` is achieved (i.e., the existing `max_accuracy` update path).
- Keep the maximum number of epochs as-is (`TRAIN.EPOCHS`, which is 50 in the base/small configs); early stopping can only end training earlier, never extend it.
- Preserve the existing `best_ckpt.pth` saving behavior on new best accuracy.
- Log a clear message when early stopping triggers, including the patience counter, the epoch at which the best accuracy was reached, and the best accuracy value.
- Make early stopping resume-safe: persist the patience counter and best-accuracy epoch alongside the existing checkpoint state (additive, backward-compatible checkpoint key) so a resumed run does not incorrectly reset its early-stopping state.
- Add a configurable `TRAIN.PATIENCE` (default 10) option; do not change optimizer, LR scheduler, model architecture, EMA, dataset, or existing checkpoint key formats.

## Capabilities

### New Capabilities
- `early-stopping`: Stop training based on validation Acc@1 with a configurable patience window, including resume-safe patience tracking and logging.

### Modified Capabilities
<!-- No existing specs in openspec/specs/ are affected. -->

## Impact

- `classification/main.py` — training loop in `main()` gains patience tracking and an early-stop break path; `max_accuracy`/validation logic is reused, not duplicated.
- `classification/utils/utils.py` — `save_checkpoint_ema` / `load_checkpoint_ema` gain an optional, backward-compatible persistence of the early-stopping patience state.
- `classification/config.py` (and `classification/configs/DAMamba/*.yaml`) — new optional `TRAIN.PATIENCE` config value (default 10).
- Existing checkpoint files remain loadable; new key is optional and absent from old checkpoints.