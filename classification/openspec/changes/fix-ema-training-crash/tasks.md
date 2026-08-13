## 1. Fix the MESA soft-target loss (crash)

- [x] 1.1 In `classification/main.py`, add `import torch.nn.functional as F` in the imports section.
- [x] 1.2 In `train_one_epoch` (main.py ~line 457), replace `ema_loss = criterion(outputs, ema_output) * mesa` with a detached soft-target KL divergence: `ema_loss = F.kl_div(F.log_softmax(outputs.float(), dim=-1), ema_output.float(), reduction='batchmean') * mesa`. Do NOT cast `ema_output` to `long`; do NOT remove the `softmax(dim=-1).detach()` that precedes it.
- [x] 1.3 Confirm the surrounding `if mesa > 0.0:` blocks (loss composition `criterion(outputs, targets) + ema_loss`) remain unchanged.

## 2. Sync EMA from loaded pretrained weights (0% EMA fix)

- [x] 2.1 In `main.py` immediately after `load_pretrained_ema(config, model_without_ddp, logger, model_ema)` (~line 259), add a one-time startup sync: `model_ema.ema.load_state_dict(model_without_ddp.state_dict())` guarded by `if model_ema is not None:`, with an INFO log line. Do NOT modify `utils/utils.py::load_pretrained_ema`.
- [x] 2.2 Confirm the online update `model_ema.update(model)` at the accumulation boundary (main.py ~line 473) is preserved unchanged.

## 3. Static verification

- [x] 3.1 Verify no `gather()` crash: with a fabricated `mesa=1.0` path and a dummy batch, KL divergence computes without dtype/index errors under AMP autocast.
- [x] 3.2 Verify EMA parity after sync: immediately after the startup sync, student and EMA produce identical predictions on a single validation batch (same argmax/accuracy), confirming the EMA no longer reports 0% when the student reports ~80.7%.
- [x] 3.3 Confirm no changes outside `classification/main.py` (no config, dataset, model-architecture, or checkpointing changes).