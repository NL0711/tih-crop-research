import os
import sys
# Insert classification directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_config
from models import build_model
from data import build_loader
from utils.utils import load_checkpoint_ema

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='classification/configs/DAMamba/damamba_tiny.yaml')
    parser.add_argument('--data-path', type=str, default='../Cauliflower_split')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--opts', default=None, nargs='+')
    
    # We need these so get_config doesn't fail
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
    
    # Override num_classes to 5
    config.defrost()
    config.MODEL.NUM_CLASSES = 5
    config.freeze()
    
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader(config)
    
    print("Building model...")
    model = build_model(config)
    model.cuda()
    
    from timm.utils import ModelEma
    model_ema = ModelEma(model, decay=0.9999)
    
    # Load checkpoint
    ckpt_path = "output/tiny/finetune/latest_ckpt.pth"
    print(f"Loading checkpoint {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    
    # Load model and model_ema
    model.load_state_dict(checkpoint['model'])
    model_ema.ema.load_state_dict(checkpoint['model_ema'])
    
    model.eval()
    model_ema.ema.eval()
    
    print("\nEvaluating on validation set...")
    student_correct = 0
    ema_correct = 0
    total = 0
    
    first_batch = True
    
    with torch.no_grad():
        for images, targets in data_loader_val:
            images = images.cuda()
            targets = targets.cuda()
            
            out_s = model(images)
            out_e = model_ema.ema(images)
            
            pred_s = out_s.argmax(dim=-1)
            pred_e = out_e.argmax(dim=-1)
            
            student_correct += (pred_s == targets).sum().item()
            ema_correct += (pred_e == targets).sum().item()
            total += targets.size(0)
            
            if first_batch:
                first_batch = False
                print("\nFirst batch details:")
                print("Targets:  ", targets.cpu().tolist())
                print("Student:  ", pred_s.cpu().tolist())
                print("EMA pred: ", pred_e.cpu().tolist())
                print("\nStudent outputs (first 5):")
                print(out_s[:5].cpu())
                print("\nEMA outputs (first 5):")
                print(out_e[:5].cpu())
                
    print(f"\nStudent validation accuracy: {student_correct}/{total} ({100.0 * student_correct / total:.3f}%)")
    print(f"EMA validation accuracy:     {ema_correct}/{total} ({100.0 * ema_correct / total:.3f}%)")

if __name__ == '__main__':
    main()
