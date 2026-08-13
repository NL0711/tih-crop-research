## Why

The CNN-hybrid DAMamba fine-tune crashes during training at `ema_loss = criterion(outputs, ema_output) * mesa` (main.py:457) with `RuntimeError: gather(): Expected dtype int64 for index`. The EMA teacher's softmax probabilities (`ema_output`, float32 `[B,5]`) are passed as the *target* of a class-index cross-entropy criterion, a type/semantic mismatch. Separately, the EMA evaluation immediately reports 0% while the normal model reports 80.7% before training, so the EMA model's initialization from the loaded 5-class pretrained weights must be verified.

## What Changes

- **classification/main.py — `train_one_epoch` (line 457):** Replace the class-index `criterion(outputs, ema_output)` with a detached soft-target distillation loss. The EMA model is an intentional teacher (config `TRAIN.MESA`, staged on from 25% of epochs), so the correct loss is KL divergence between the student's `log_softmax` and the EMA's detached softmax probabilities, weighted by `mesa`. No plain `long()` cast of `ema_output`.
- **classification/main.py — EMA initialization (after `load_pretrained_ema`, line 259):** Guarantee the EMA model starts from the freshly-loaded 5-class model weights (the EMA is deep-copied at main.py:171 *before* pretrained loading; sync it from the loaded model so its immediate evaluation can never be a 0% outlier). Does not modify `load_pretrained_ema` itself.
- **Verification only — no changes** to class-count mapping, CNN/hybrid architecture, pretrained-loading function, LR/scheduler/optimizer/epochs/augmentation, or dataset handling. No full training run.

## Capabilities

### New Capabilities

- `ema-soft-target-distillation`: The MESA EMA-as-teacher training term uses a detached soft-target KL-divergence loss (student `log_softmax` vs EMA `softmax` probabilities), is multiplied by `mesa`, and is added to the standard classification loss — never feeding float probabilities into a class-index criterion.
- `ema-initialization-sync`: The EMA model is initialized from the loaded 5-class pretrained model weights at startup so the immediate (pre-training) EMA evaluation reflects the same weights as the normal model, and EMA update-from-model during training is preserved.

### Modified Capabilities

<!-- None: no main specs exist yet. -->

## Impact

- `classification/main.py` — MESA block in `train_one_epoch` (line 452-460) and the EMA init step after `load_pretrained_ema` (line ~259); adds `torch.nn.functional` import.
- No changes to `classification/utils/utils.py`, `models/`, `configs/`, `data/`, or the dataset.
- Downstream effects: MESA-enabled runs no longer crash from epoch 25% onward; `ema_loss` becomes a proper bounded soft-target term; EMA pre-training eval matches the model's (no spurious 0%).
