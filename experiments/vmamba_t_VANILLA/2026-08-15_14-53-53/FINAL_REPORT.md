# DAMamba Experiment – Final Report

**Generated:** 2026-08-15 15:34:07

---

## Experiment

| Key | Value |
|-----|-------|
| Name | vmamba_t_VANILLA |
| Timestamp | 2026-08-15_14-53-53 |
| Architecture | vmamba_t |
| Model | vmamba_t |
| Hybrid enabled | False |
| Output dir | ./experiments\vmamba_t_VANILLA\2026-08-15_14-53-53 |

## Dataset

| Split | Samples |
|-------|---------|
| Train | 1820 |
| Val   | 135 |
| Test  | 143 |
| Classes | 5 |

## Training Configuration

| Param | Value |
|-------|-------|
| Epochs | 50 |
| Batch size | 32 |
| Base LR | 0.000125 |
| Weight decay | 0.05 |
| Optimizer | adamw |
| Scheduler | cosine |
| Warmup epochs | 5 |
| AMP | True |
| Seed | 0 |
| Total training time | 0:39:21 |

## Validation

| Metric | Value |
|--------|-------|
| Best Val Acc@1 | **50.37%** |
| Best Epoch | 2 |
| Final Val Acc@1 | 49.6296297426577 |
| Final Val Acc@5 | 100.00000039559824 |

## Test

_Run evaluate_test.py or use --eval flag to populate test metrics._

## Plots

See plots/ subdirectory.

## Reproducibility

| Item | Value |
|------|-------|
| Command | .\classification\main.py --cfg .\classification\configs\models\vmamba_tiny.yaml --data-path C:\Users\blais\Downloads\caulieval\Cauliflower_split --output output/benchmark/vmamba-t --no-subdir |
| Git commit | 2525870 |
| Git branch | resnet |
| Seed | 0 |
| Config | ./experiments\vmamba_t_VANILLA\2026-08-15_14-53-53\config.yaml |
