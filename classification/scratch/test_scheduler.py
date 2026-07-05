import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config import get_config
from utils.lr_scheduler import build_scheduler

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='classification/configs/DAMamba/damamba_tiny.yaml')
    parser.add_argument('--data-path', type=str, default='../Cauliflower_split')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--opts', default=None, nargs='+')
    
    # Dummy args
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part')
    parser.add_argument('--pretrained')
    parser.add_argument('--resume')
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--disable_amp', action='store_true')
    parser.add_argument('--output', default='output')
    parser.add_argument('--tag')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--oversample', action='store_true')
    parser.add_argument('--fused_layernorm', action='store_true')
    parser.add_argument('--optim')
    parser.add_argument('--model_ema', type=bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', type=bool, default=False)
    parser.add_argument('--memory_limit_rate', type=float, default=-1)
    parser.add_argument('--ddp', type=str, default='torch')
    parser.add_argument('--enable_preload', action='store_true')
    parser.add_argument('--enable_persistance', action='store_true')
    parser.add_argument('--mute_repeat', action='store_true')
    
    args = parser.parse_args([])
    config = get_config(args)
    
    # Let's construct a dummy optimizer with 1 parameter
    p = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([p], lr=config.TRAIN.BASE_LR)
    
    n_iter_per_epoch = 34
    lr_scheduler = build_scheduler(config, optimizer, n_iter_per_epoch)
    
    print("Scheduler params:")
    print("t_initial:", lr_scheduler.t_initial)
    print("warmup_t:", lr_scheduler.warmup_t)
    print("lr_min:", lr_scheduler.lr_min)
    print("warmup_lr_init:", lr_scheduler.warmup_lr_init)
    
    # Get LR at some steps
    for epoch in [0, 1, 4, 5, 10, 15, 50, 99]:
        step = epoch * n_iter_per_epoch
        lrs = lr_scheduler._get_lr(step)
        print(f"Epoch {epoch:2d} (step {step:4d}): lr = {lrs[0]:.8e}")

if __name__ == '__main__':
    main()
