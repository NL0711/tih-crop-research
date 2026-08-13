## Context

DAMamba-T is fine-tuned on a 5-class cauliflower disease dataset. The pipeline (Swin-based) has two latent bugs:

1. `data/build.py` `build_dataset()` returns a hardcoded `nb_classes = 1000` for the `imagenet`/ImageFolder path. `build_loader()` (build.py:112) assigns that to `config.MODEL.NUM_CLASSES`, running **before** `build_model()`. So even with `MODEL.NUM_CLASSES=5` in the CLI opts, the model is built with a 1000-way head. The saved `config.json` shows `5` only because it is dumped before `build_loader` mutates the config.
2. `models/__init__.py` `build_model()` forwards only 8 DAMamba kwargs and drops `config.MODEL.HYBRID.ENABLE`. `hybrid_enable` stays `False` (DAMamba.py:636), so `local_cnn`/`fusion` are never instantiated and the CNN hybrid branch is dead code.

Consequences observed (both `cnn_hybrid` and `vanilla` logs, identical): `<All keys matched successfully>`, Loss ≈ 8.3359, Acc@1 = 0%, Acc@5 = 0%. Acc@5 = 0% is mathematically impossible for a 5-logit output (top-5 exhausts all classes ⇒ 100%), proving the head emitted 1000 ImageNet logits; the pretrained ImageNet classifier loaded wholesale and its predictions never match targets {0..4}.

Constraints: no changes to LR/optimizer/scheduler/augmentation/epochs/checkpoints/pretrained weights/dataset contents; no redesign of the hybrid internals; must stay backward compatible when hybrid is disabled.

## Goals / Non-Goals

**Goals:**
- Make `config.MODEL.NUM_CLASSES` reflect the real dataset class count before `build_model()` runs.
- Forward the existing `config.MODEL.HYBRID.ENABLE` to `DAMamba(..., hybrid_enable=...)`.
- Ensure the pretrained 1000-class head is never loaded into the 5-class head; load only compatible backbone weights (targeted, without weakening global checkpoint validation).
- Preserve backward compatibility when hybrid is disabled.
- Keep changes minimal and surgical (build.py, models/__init__.py, and the targeted filter in utils/utils.py).

**Non-Goals:**
- No changes to DAMamba internals (`LocalCNN`, `HybridFusion`, `Dynamic_Adaptive_Scan`, DCNv3, selective-scan, `MlpHead`).
- No hyperparameter, augmentation, or training-loop changes.
- No changes to checkpoint save/load semantics beyond what already exists.
- No new configuration keys.

## Decisions

### D1: Derive class count from the actual dataset
Replace the hardcoded `nb_classes = 1000` in the ImageFolder path of `build_dataset()` with `nb_classes = len(dataset.classes)`. `dataset.classes` is the sorted list of class names assigned by `torchvision.datasets.ImageFolder`, which is exactly the space over which the model's targets range. This fixes the class count for any folder dataset shape without special-casing `5` or `1000`.

- **Alternative considered**: reading `config.MODEL.NUM_CLASSES` instead of overwriting it in `build_loader`. Rejected — the ImageNet path elsewhere needs the real count (ImageNet-22k sets its own 21841), and the "derive from the actual data" approach is single-source-of-truth.
- **Alternative considered**: deriving from `len(dataset.class_to_idx)`. Equivalent to `len(dataset.classes)`; the latter is used since `main.py` already reads `dataset.classes` for class names.

Ordering note: this still runs before `build_model()` (main.py:132 → 135), so the derived value is what `build_model` consumes.

### D2: Forward the existing hybrid flag
In `build_model()`, add `hybrid_enable=config.MODEL.HYBRID.ENABLE` to the `DAMamba(...)` call. `_C.MODEL.HYBRID.ENABLE` already exists (config.py:224-225, default `False`), so no new key is introduced. With `True`, the constructor instantiates `local_cnn` + `fusion` (randomly initialized via `self._init_weights`); the pretrained checkpoint does not contain these keys, so `strict=False` leaves them at initialization. With `False`, the constructor path is identical to today.

### D3: Handle the 1000→5 classifier mismatch at load time
The naive assumption — `strict=False` silently skips shape-mismatched head keys — does not hold on the pinned `torch==2.2.0`. In torch 2.2+, `Module.load_state_dict` records shape mismatches as `error_msgs` inside `_load_from_state_dict` and then raises `RuntimeError` whenever `error_msgs` is non-empty, **regardless of `strict`**. (The `if strict:` block only promotes missing/unexpected keys into errors; size mismatches always raise.) My probe on torch 2.12 confirms this: a `[1000,512]` head loaded into a `[5,512]` model raises even with `strict=False`.

The correct targeted fix: in `utils/utils.py::load_pretrained_ema`, pre-filter the checkpoint state dict, dropping keys whose tensor shape is incompatible with the target model (only the classifier head keys `head.fc.weight`/`head.fc.bias` in this case), then call `model.load_state_dict(filtered, strict=False)`. This:
- Loads all compatible backbone weights.
- Leaves the incompatible head at its random initialization.
- Does **not** weaken global validation: still strict for names (missing/unexpected reported), only shape-incompatible keys are removed; verify via `_IncompatibleKeys`/size-mismatch log lines instead of suppressing them.
- Avoids the unconditional raise on torch 2.2+.

- **Alternative considered**: calling `load_state_dict(..., strict=True)` and surgically popping the head keys. Same outcome, but relies on raising first; the pre-filter is cleaner and version-robust across torch 2.0–2.12.
- **Alternative considered**: leaving utils.py untouched and relying on `strict=False`. Rejected — crashes on torch 2.2+ once the head is 5-dim.

## Risks / Trade-offs

- [If any run still expects 1000 logits (e.g., legacy eval scripts)` → 5-logit output is the intended correct behavior; verify with the model's `head.fc.weight.shape`.
- [Hybrid path first-load loss will be higher than vanilla because `local_cnn`/`fusion` start untrained] → Expected; only a real training run populates them, which is out of scope for this change.
- [`MODEL.NUM_CLASSES` semantics now follow the data rather than user opts] → This is the desired behavior; the CLI value that disagrees with the dataset is silently corrected by the derived count, and all splits share the same mapping so targets stay in `[0, num_classes)`.
- [Filtering shape-incompatible keys could mask a genuinely corrupted checkpoint] → The filter only drops keys whose shapes cannot possibly load; name-level missing/unexpected keys are still reported and validated. Corrupted-write checks (tensor shapes for every copied key) remain enforced by `load_state_dict`.
- [Resume path (`load_checkpoint_ema`) not modified] → Intentionally unchanged; resuming a mismatched checkpoint should still surface loudly. Only the `PRETRAINED` fine-tune path filters incompatible classifier keys.

## Migration Plan

1. Apply the three code edits (build.py, models/__init__.py, utils/utils.py).
2. Static verification: import `build_dataset` against a small folder fixture or inspect the existing cauliflower layout to confirm `len(dataset.classes) == 5`; confirm `build_model` kwargs; confirm `head.fc` output dim is 5; confirm the load filter drops only shape-incompatible head keys.
3. A single inference pass (no training) may be run by the user to confirm output shape `[B, 5]`, targets in `[0, 5)`, hybrid flag state, and that backbone weights loaded while `head.fc` stayed at init.
4. Rollback: revert `data/build.py` and `models/__init__.py` (two-line change each) and `utils/utils.py` (targeted filter).

## Open Questions

- None blocking; the corrected behavior (5-logit output, hybrid flag forwarding, safe classifier load, untouched internals) is fully determined by the three edits.