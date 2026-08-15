# DAMamba Experiment – Final Report

**Generated:** 2026-08-15 13:44:39

---

## Experiment

| Key | Value |
|-----|-------|
| Name | resnet50_VANILLA |
| Timestamp | 2026-08-15_12-10-47 |
| Architecture | resnet50 |
| Model | resnet50 |
| Hybrid enabled | False |
| Output dir | ./experiments\resnet50_VANILLA\2026-08-15_12-10-47 |

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
| Batch size | 64 |
| Base LR | 0.00125 |
| Weight decay | 0.0001 |
| Optimizer | sgd |
| Scheduler | cosine |
| Warmup epochs | 5 |
| AMP | True |
| Seed | 0 |
| Total training time | 1:32:59 |

## Validation

| Metric | Value |
|--------|-------|
| Best Val Acc@1 | **91.11%** |
| Best Epoch | 21 |
| Final Val Acc@1 | 91.11111150670935 |
| Final Val Acc@5 | 100.00000039559824 |

## Test

_Run evaluate_test.py or use --eval flag to populate test metrics._

## Plots

See plots/ subdirectory.

## Reproducibility

| Item | Value |
|------|-------|
| Command | .\classification\main.py --cfg .\classification\configs\models\resnet50.yaml --data-path C:\Users\blais\Downloads\caulieval\Cauliflower_split --output output/benchmark/resnet50 --no-subdir |
| Git commit | 2dd14d9 |
| Git branch | main |
| Seed | 0 |
| Config | ./experiments\resnet50_VANILLA\2026-08-15_12-10-47\config.yaml |
