# DAMamba Experiment – Final Report

**Generated:** 2026-08-16 09:50:04

---

## Experiment

| Key | Value |
|-----|-------|
| Name | inception_resnet_v2_VANILLA |
| Timestamp | 2026-08-16_09-03-45 |
| Architecture | inception_resnet_v2 |
| Model | inception_resnet_v2 |
| Hybrid enabled | False |
| Output dir | ./experiments\inception_resnet_v2_VANILLA\2026-08-16_09-03-45 |

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
| Total training time | 0:45:56 |

## Validation

| Metric | Value |
|--------|-------|
| Best Val Acc@1 | **90.37%** |
| Best Epoch | 10 |
| Final Val Acc@1 | 88.88888928448712 |
| Final Val Acc@5 | 100.00000039559824 |

## Test

_Run evaluate_test.py or use --eval flag to populate test metrics._

## Plots

See plots/ subdirectory.

## Reproducibility

| Item | Value |
|------|-------|
| Command | .\classification\main.py --cfg .\classification\configs\models\inception_resnet_v2.yaml --data-path C:\Users\blais\Downloads\caulieval\Cauliflower_split --output output/benchmark/InceptioResNetV2 --no-subdir |
| Git commit | 2525870 |
| Git branch | resnet |
| Seed | 0 |
| Config | ./experiments\inception_resnet_v2_VANILLA\2026-08-16_09-03-45\config.yaml |
