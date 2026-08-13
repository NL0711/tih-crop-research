## Context

The DAMamba-T CNN-hybrid fine-tune (5-class cauliflower disease dataset) crashes in `train_one_epoch` when MESA activates (from epoch 25% onward, `main.py:330`):

```python
ema_output = model_ema.ema(samples).detach()      # [B,5] logits
ema_output = ema_output.softmax(dim=-1).detach()  # [B,5] float32 probabilities
ema_loss = criterion(outputs, ema_output) * mesa  # CRASH
```

`criterion` is `torch.nn.CrossEntropyLoss()` (no mixup/label-smoothing in the active config), which gathers the target as a class index. A float probability tensor as target hits `gather(): Expected dtype int64 for index`. (`timm.LabelSmoothingCrossEntropy` and `SoftTargetCrossEntropy` would also be fed mismatched inputs; only `SoftTargetCrossEntropy` accepts float targets, and it is not the active criterion.)

Separately the log/behavior shows the EMA model evaluating at 0% while the student evaluates at ~80.7% before training (e.g., `output-test/tiny/cnn_hybrid/log_rank0.txt`).

Constraints: do NOT change class-count mapping, CNN/hybrid architecture, pretrained loading function, LR/optimizer/scheduler/epochs/augmentation, or dataset handling. Do NOT run a full training experiment.

## Goals / Non-Goals

**Goals:**
- Remove the `gather(): Expected dtype int64 for index` crash with a correct soft-target distillation loss (KL divergence) consistent with the existing EMA-as-teacher design.
- Guarantee the EMA model starts from the loaded 5-class pretrained weights so its pre-training evaluation is meaningful (matches the student).
- Preserve EMA update (`model_ema.update(model)`) and EMA eval/checkpointing behavior.
- Minimal, surgical edits confined to `classification/main.py`.

**Non-Goals:**
- No changes to `utils/utils.py` (`load_pretrained_ema`, `load_checkpoint_ema`, `save_checkpoint_ema`).
- No changes to the CNN/hybrid modules, config files, LR, scheduler, epochs, augmentation, or dataset.
- No full training run; only static verification (inference sanity, optional single-batch).

## Decisions

### D1: MESA is a soft-target (EMA-as-teacher) distillation loss — use KL divergence
- **Rationale**: The design intent in the DAMamba lineage (`mesa` = config `TRAIN.MESA`, e.g. 1.0/1.5/2.0, staged on from `int(0.25 * EPOCHS)`; `ema_output` explicitly softmaxed and detached) is a teacher-consistency/self-distillation term where the EMA model provides soft targets. 
- Replace line 457 with:
  ```python
  ema_loss = F.kl_div(
      F.log_softmax(outputs.float(), dim=-1),
      ema_output.float(),
      reduction='batchmean') * mesa
  ```
  This is the correct soft-target distillation loss: it minimizes the KL divergence from the student's log-softmax toward the EMA teacher's probability distribution. `ema_output` is already detached. The `.float()` guards against dtype mismatch under AMP autocast. Add `import torch.nn.functional as F` at top of `main.py`.
- **Alternative considered (rejected)**: `torch.clamp`/rounding or `ema_output.argmax()` to synthesize hard labels — this discards the soft teacher signal and violates the "do not simply cast to long" requirement.
- **Alternative considered (rejected)**: remove/disable the EMA loss entirely (branch 4 of the requirement). Rejected because the code clearly *intends* a distillation term (dedicated config key, staged activation, explicitly detached softmax). Disabling it would silently delete an intended feature.
- **Alternative considered (`criterion = SoftTargetCrossEntropy` hack)**: changing `criterion` selection so `criterion(outputs, ema_output)` works. Rejected — it couples the classification criterion to the distillation term and would alter the standard loss path for `targets` under mixup.

### D2: Sync EMA from the loaded student model after `load_pretrained_ema`
- **Root cause of the 0% EMA**: `model_ema = ModelEma(model, ...)` is constructed at `main.py:171` via `deepcopy`, i.e. **before** pretrained weights are loaded into the model at `main.py:258-259`. `load_pretrained_ema` separately loads `checkpoint['model_ema']`/`checkpoint['model']` into the EMA (`utils/utils.py:93-103`), but when the pretrained checkpoint's `model_ema` entry is stale, missing the 5-class/hybrid keys, or shape-incompatible, those keys are dropped by `_filter_shape_incompatible` and the EMA keeps its pre-load (random) head/CNN-fusion init while the student is loaded correctly — producing the observed 80.7% student vs 0% EMA divergence.
- **Fix**: immediately after `load_pretrained_ema(config, model_without_ddp, logger, model_ema)` in `main.py:259`, add:
  ```python
  if model_ema is not None:
      model_ema.ema.load_state_dict(model_without_ddp.state_dict())
      logger.info("EMA initialized from loaded pretrained model weights")
  ```
  This forces the EMA to be an exact snapshot of the correctly-loaded 5-class model. It does not modify `load_pretrained_ema`, does not change the online `model_ema.update(model)` loop, and only affects the one-time startup state.
- **Alternative considered (rejected)**: fixing the EMA-init logic inside `load_pretrained_ema`. Rejected per the explicit constraint that pretrained-loading code must not change; the sync belongs at the call site in `main.py`.

### D3: Preserve online EMA update
- Keep `model_ema.update(model)` at the gradient-accumulation boundary (`main.py:473-474`) and all EMA eval calls / checkpointing unchanged. The D1/D2 edits touch only the loss term and startup init.

## Risks / Trade-offs

- [KL-divergence changes loss scale vs. the original soft-target cross-entropy intent] → `reduction='batchmean'` averages over the batch (matches CE's batch-wise mean); the `mesa` weight is preserved. Numerical behavior is the standard distillation loss.
- [Repo/EMA logits under AMP are fp16] → `.float()` on both inputs normalizes dtype; `ema_output` is detached so no gradients flow through it.
- [Forcing EMA snapshot at startup could mask an intentionally useful stored EMA] → For transfer-learning from ImageNet to a 5-class head, the student snapshot is the canonical starting point; this is what the sync guarantees. The old `model_ema` state is still saved/loadable via checkpoints.
- [Verification only, no training run] → correctness is confirmed by static introspection (dtype/shape), an optional single-batch forward of both models, plus the existing pre-training validation line.

## Migration Plan

1. Edit `classification/main.py`: add `import torch.nn.functional as F`; replace the MESA `ema_loss` line; add the EMA startup sync after `load_pretrained_ema`.
2. Static verification: import-mount the module, run a single batch through `model` and `model_ema.ema`, assert no `gather` error when `mesa=1.0`; assert EMA weights equal student weights right after the sync.
3. No training run. Rollback = revert the two edited regions in `main.py`.

## Open Questions

- None blocking. The intended EMA behavior is confirmed to be a soft-target distillation/consistency teacher (not merely an eval checkpoint), so the KL-divergence fix (option 3 in the requirements) is selected over disabling the term.