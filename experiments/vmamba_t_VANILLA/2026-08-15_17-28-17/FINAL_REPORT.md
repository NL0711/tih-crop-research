# DAMamba Experiment – Final Report

**Generated:** 2026-08-15 18:02:01

---

## Experiment

| Key | Value |
|-----|-------|
| Name | vmamba_t_VANILLA |
| Timestamp | 2026-08-15_17-28-17 |
| Architecture | vmamba_t |
| Model | vmamba_t |
| Hybrid enabled | False |
| Output dir | ./experiments\vmamba_t_VANILLA\2026-08-15_17-28-17 |

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
| Epochs | 25 |
| Batch size | 32 |
| Base LR | 3.125e-05 |
| Weight decay | 0.05 |
| Optimizer | adamw |
| Scheduler | cosine |
| Warmup epochs | 2 |
| AMP | True |
| Seed | 0 |
| Total training time | 0:33:22 |

## Validation

| Metric | Value |
|--------|-------|
| Best Val Acc@1 | **37.04%** |
| Best Epoch | 7 |
| Final Val Acc@1 | 25.185185185185187 |
| Final Val Acc@5 | 100.00000039559824 |

## Test

_Run evaluate_test.py or use --eval flag to populate test metrics._

## Plots

See plots/ subdirectory.

## Reproducibility

| Item | Value |
|------|-------|
| Command | .\classification\main.py --cfg .\classification\configs\models\vmamba_tiny.yaml --data-path C:\Users\blais\Downloads\caulieval\Cauliflower_split --output output/benchmark/vmamba_t --no-subdir |
| Git commit | 2525870 |
| Git branch | resnet |
| Seed | 0 |
| Config | ./experiments\vmamba_t_VANILLA\2026-08-15_17-28-17\config.yaml |
