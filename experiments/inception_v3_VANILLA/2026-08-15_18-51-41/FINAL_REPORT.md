# DAMamba Experiment – Final Report

**Generated:** 2026-08-15 20:43:58

---

## Experiment

| Key | Value |
|-----|-------|
| Name | inception_v3_VANILLA |
| Timestamp | 2026-08-15_18-51-41 |
| Architecture | inception_v3 |
| Model | inception_v3 |
| Hybrid enabled | False |
| Output dir | ./experiments\inception_v3_VANILLA\2026-08-15_18-51-41 |

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
| Total training time | 1:51:56 |

## Validation

| Metric | Value |
|--------|-------|
| Best Val Acc@1 | **92.59%** |
| Best Epoch | 41 |
| Final Val Acc@1 | 91.11111150670935 |
| Final Val Acc@5 | 100.00000039559824 |

## Test

_Run evaluate_test.py or use --eval flag to populate test metrics._

## Plots

See plots/ subdirectory.

## Reproducibility

| Item | Value |
|------|-------|
| Command | .\classification\main.py --cfg .\classification\configs\models\inception_v3.yaml --data-path C:\Users\blais\Downloads\caulieval\Cauliflower_split --output output/benchmark/inception_v3 --no-subdir |
| Git commit | 2525870 |
| Git branch | resnet |
| Seed | 0 |
| Config | ./experiments\inception_v3_VANILLA\2026-08-15_18-51-41\config.yaml |
