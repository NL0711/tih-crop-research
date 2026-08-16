# DAMamba Experiment – Final Report

**Generated:** 2026-08-15 14:40:14

---

## Experiment

| Key | Value |
|-----|-------|
| Name | vmamba_t_VANILLA |
| Timestamp | 2026-08-15_14-32-57 |
| Architecture | vmamba_t |
| Model | vmamba_t |
| Hybrid enabled | False |
| Output dir | ./experiments\vmamba_t_VANILLA\2026-08-15_14-32-57 |

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
| Epochs | 2 |
| Batch size | 32 |
| Base LR | 0.000125 |
| Weight decay | 0.05 |
| Optimizer | adamw |
| Scheduler | cosine |
| Warmup epochs | 1 |
| AMP | True |
| Seed | 0 |
| Total training time | 0:06:59 |

## Validation

| Metric | Value |
|--------|-------|
| Best Val Acc@1 | **44.44%** |
| Best Epoch | 1 |
| Final Val Acc@1 | 44.44444444444444 |
| Final Val Acc@5 | 100.00000039559824 |

## Test

_Run evaluate_test.py or use --eval flag to populate test metrics._

## Plots

See plots/ subdirectory.

## Reproducibility

| Item | Value |
|------|-------|
| Command | c:\Users\blais\Desktop\Full-Stack-Projects\tih-crop-research\classification\main.py --cfg c:\Users\blais\Desktop\Full-Stack-Projects\tih-crop-research\classification\configs\models\vmamba_tiny.yaml --data-path C:\Users\blais\Downloads\caulieval\Cauliflower_split --output c:\Users\blais\Desktop\Full-Stack-Projects\tih-crop-research\output\benchmark\vmamba_t --no-subdir --opts TRAIN.EPOCHS 2 TRAIN.WARMUP_EPOCHS 1 TRAIN.PATIENCE 999 DATA.NUM_WORKERS 0 --model_ema False |
| Git commit | 2525870 |
| Git branch | resnet |
| Seed | 0 |
| Config | ./experiments\vmamba_t_VANILLA\2026-08-15_14-32-57\config.yaml |
