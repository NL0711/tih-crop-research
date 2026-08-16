# DAMamba Experiment – Final Report

**Generated:** 2026-08-15 22:37:49

---

## Experiment

| Key | Value |
|-----|-------|
| Name | efficientnet_b1_VANILLA |
| Timestamp | 2026-08-15_21-18-13 |
| Architecture | efficientnet_b1 |
| Model | efficientnet_b1 |
| Hybrid enabled | False |
| Output dir | ./experiments\efficientnet_b1_VANILLA\2026-08-15_21-18-13 |

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
| Base LR | 3.125e-05 |
| Weight decay | 0.05 |
| Optimizer | adamw |
| Scheduler | cosine |
| Warmup epochs | 2 |
| AMP | True |
| Seed | 0 |
| Total training time | 1:19:09 |

## Validation

| Metric | Value |
|--------|-------|
| Best Val Acc@1 | **93.33%** |
| Best Epoch | 22 |
| Final Val Acc@1 | 90.3703704268844 |
| Final Val Acc@5 | 100.00000039559824 |

## Test

_Run evaluate_test.py or use --eval flag to populate test metrics._

## Plots

See plots/ subdirectory.

## Reproducibility

| Item | Value |
|------|-------|
| Command | .\classification\main.py --cfg .\classification\configs\models\efficientnet_b1.yaml --data-path C:\Users\blais\Downloads\caulieval\Cauliflower_split --output output/benchmark/EfficientNetB1 --no-subdir |
| Git commit | 2525870 |
| Git branch | resnet |
| Seed | 0 |
| Config | ./experiments\efficientnet_b1_VANILLA\2026-08-15_21-18-13\config.yaml |
