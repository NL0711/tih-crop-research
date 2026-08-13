## Context

`classification/main.py` is a Swin Transformer-style training script adapted for DAMamba. The training loop in `main()` (lines ~326–405) already tracks a running best validation accuracy (`max_accuracy`, defaulting to `0.0`), updates it each epoch when `acc1 > max_accuracy`, and saves `best_ckpt.pth` on every improvement. On resume, `utils/utils.py::load_checkpoint_ema` restores `max_accuracy` (and `max_accuracy_ema`) from the checkpoint. There is currently no early-stopping: training always runs to `TRAIN.EPOCHS` (50 for base/small configs).

Goal: stop training when validation Acc@1 has not improved for `PATIENCE = 10` consecutive epochs, reusing the existing `max_accuracy`/validation logic and saving behavior.

## Goals / Non-Goals

**Goals:**
- Early-stop on validation `acc1` after `PATIENCE` consecutive epochs without improvement (patience reset whenever a new best `acc1` is achieved).
- Preserve existing `best_ckpt.pth` saving, `latest_ckpt.pth` saving, EMA logic, and post-training test evaluation.
- Make early-stopping resume-safe (patience counter and best-accuracy epoch persist across checkpoint resumes).
- Log the patience counter, best epoch, and best accuracy when early stopping triggers.
- Add a configurable `TRAIN.PATIENCE` (default 10).

**Non-Goals:**
- Do NOT change optimizer, LR scheduler, model architecture, EMA, dataset, or the max training epochs (`TRAIN.EPOCHS` stays 50).
- Do NOT change existing checkpoint key formats or break compatibility with old checkpoints.
- No CLI flag for patience (config-driven only).
- No changes to the EMA-based `best_ckpt_ema` behavior beyond passing shared state.

## Decisions

### 1. Integrate into the existing `max_accuracy` block, no duplicate validation
Track `patience` and `best_acc_epoch` local variables in `main()`, updated in the existing `if acc1 > max_accuracy:` block. On improvement: `patience = 0`, `best_acc_epoch = epoch`. Otherwise: `patience += 1`. When `patience >= PATIENCE`, log and `break`. This reuses the already-computed `acc1` — no second `validate()` call.
- *Alternative considered:* a separate `EarlyStopping` class in its own module. Rejected: the state is tiny, and keeping it inline next to `max_accuracy` avoids duplicate logic and a new file, matching the requirement to "integrate with existing max_accuracy/validation logic".

### 2. Config-driven patience via `_C.TRAIN.PATIENCE = 10`
Add the default in `config.py`. The YAML configs (`configs/DAMamba/*.yaml`) override only what they set, so no YAML edit is required; `TRAIN.EPOCHS` is left untouched.
- *Alternative considered:* hardcoding 10. Rejected: the codebase is config-driven, so a config value is idiomatic.

### 3. Resume-safe patience via additive, backward-compatible checkpoint keys
`max_accuracy` is already persisted. To satisfy "don't reset the best-accuracy state incorrectly on resume", persist the patience state as two **optional** keys in the checkpoint dict: `early_stop_patience` and `best_acc_epoch`.
- `save_checkpoint_ema` gains optional keyword params (`patience=0`, `best_acc_epoch=-1`) and writes the keys when provided.
- `load_checkpoint_ema` reads them with safe defaults (`patience=0`, `best_acc_epoch=-1` → use `TRAIN.START_EPOCH` on main side) when absent.
- Existing keys (`model`, `optimizer`, `lr_scheduler`, `max_accuracy`, `scaler`, `epoch`, `config`, `steps`, `model_ema`, `max_accuray_ema`) are unchanged, so old checkpoints still load and new checkpoints load in old code.
- All three save sites (`latest_ckpt`, `best_ckpt`, `best_ckpt_ema`) pass the same shared patience state so `auto_resume_helper` (which picks the most recently modified `.pth`, i.e. `latest_ckpt`) restores correct state.
- *Alternative considered:* recomputing patience from scratch (start at 0) on every resume. Rejected: it would grant a fresh 10-epoch window after every resume, defeating early stopping's purpose and only partially satisfying the resume requirement. The additive-key approach keeps format compatibility while persisting real state. If strict "no checkpoint modification at all" is later required, this is the fallback.

### 4. Early-stop check at the end of the loop body, `break` out
Place the patience check after the existing EMA validation, TensorBoard, and tracker logging (end of the epoch body). `break` exits the loop; the code after the loop (`writer.close()`, post-training test evaluation, `tracker.on_training_complete`) still runs. This ensures the final epoch's EMA metrics and tracking are logged before stopping, and the "trigger" log includes the final patience value.
- *Alternative considered:* checking immediately after the `acc1` block. Rejected: it would skip EMA validation/tracker logging for the stopping epoch.

### 5. Rank-consistent decision (no `is_main_process` gating on the check)
`acc1` is already reduced across ranks via `reduce_tensor`, and the `max_accuracy` update runs on every rank (only saving is gated). The patience update and the `break` therefore execute identically on all ranks, so DDP stays consistent without extra synchronization.

## Risks / Trade-offs

- **Additive checkpoint keys change the dict shape** → Mitigation: keys are optional and defaulted on load; old checkpoints and old code remain compatible (no existing key is altered).
- **Resuming from a pre-change checkpoint resets patience to 0** → Mitigation: `max_accuracy` is still restored, so only the freshness of the patience window is lost (gives a fresh 10-epoch grace period). Logged behavior is deterministic and acceptable.
- **Early stop ends training before LR schedule finishes** → Mitigation: intended; scheduler state simply stops being updated. No code path depends on the schedule completing.
- **Break skips remaining epochs' `latest_ckpt` saves** → Mitigation: `latest_ckpt` for the final epoch was already saved at the top of that epoch's iteration (before validation), so the newest checkpoint and its persisted patience state exist.

## Migration Plan

No data migration. Deploy: add `_C.TRAIN.PATIENCE`, extend save/load helpers, update `main()` loop. Rollback: revert the three files; existing checkpoints load unchanged since the new keys are optional.

## Open Questions

- None blocking. Optional future: expose `--patience` CLI override (out of scope per non-goals).
