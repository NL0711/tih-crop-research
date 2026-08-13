## 1. Config

- [x] 1.1 Add `_C.TRAIN.PATIENCE = 10` to `classification/config.py` (leave `TRAIN.EPOCHS` unchanged)

## 2. Checkpoint persistence

- [x] 2.1 Extend `save_checkpoint_ema` in `classification/utils/utils.py` with optional `patience` and `best_acc_epoch` keyword params and write additive `early_stop_patience` / `best_acc_epoch` keys when provided
- [x] 2.2 Extend `load_checkpoint_ema` in `classification/utils/utils.py` to read the optional `early_stop_patience` / `best_acc_epoch` keys with safe defaults and return them

## 3. Training loop integration

- [x] 3.1 Initialize `patience` and `best_acc_epoch` state in `main()` alongside `max_accuracy`; use the values returned by `load_checkpoint_ema` on resume
- [x] 3.2 In the existing `if acc1 > max_accuracy:` block, reset patience to 0 and record `best_acc_epoch`; otherwise increment patience; pass the state to the `latest_ckpt`, `best_ckpt`, and `best_ckpt_ema` save calls
- [x] 3.3 Add the early-stop check after EMA validation and tracker logging: when `patience >= config.TRAIN.PATIENCE`, log the patience counter, best epoch, and best accuracy, then `break` the epoch loop

## 4. Verification

- [x] 4.1 Syntax-check `classification/main.py` and `classification/utils/utils.py` compile
- [x] 4.2 Review diff: confirm no changes to optimizer, LR scheduler, model, EMA, dataset, or existing checkpoint key names
