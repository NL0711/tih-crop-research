# DAMamba Experiment – Final Report

**Generated:** 2026-08-17 21:52:35

---

## Experiment

| Key | Value |
|-----|-------|
| Name | tiny_VANILLA |
| Timestamp | 2026-08-17_21-09-36 |
| Architecture | DAMamba |
| Model | tiny |
| Hybrid enabled | False |
| Output dir | ./experiments\tiny_VANILLA\2026-08-17_21-09-36 |

## Dataset

| Split | Samples |
|-------|---------|
| Train | 25704 |
| Val   | 1399 |
| Test  | 1407 |
| Classes | 7 |

## Training Configuration

| Param | Value |
|-------|-------|
| Epochs | 10 |
| Batch size | 32 |
| Base LR | 0.0001 |
| Weight decay | 0.05 |
| Optimizer | adamw |
| Scheduler | cosine |
| Warmup epochs | 1 |
| AMP | True |
| Seed | 0 |
| Total training time | 0:41:51 |

## Validation

| Metric | Value |
|--------|-------|
| Best Val Acc@1 | **97.50%** |
| Best Epoch | 8 |
| Final Val Acc@1 | 97.42673338098642 |
| Final Val Acc@5 | 100.0 |

## Test

_Run evaluate_test.py or use --eval flag to populate test metrics._

## Plots

See plots/ subdirectory.

## Reproducibility

| Item | Value |
|------|-------|
| Command | classification/main.py --cfg classification/configs/DAMamba/damamba_tiny.yaml --data-path C:\Users\blais\Downloads\caulieval\Mustard_split --output output --tag finetune |
| Git commit | 2dd14d9 |
| Git branch | main |
| Seed | 0 |
| Config | ./experiments\tiny_VANILLA\2026-08-17_21-09-36\config.yaml |
