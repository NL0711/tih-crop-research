import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config import get_config
from models import build_model
from data import build_loader

def get_predictions(ckpt_path, config, data_loader_val):
    model = build_model(config)
    model.cuda()
    from timm.utils import ModelEma
    model_ema = ModelEma(model, decay=0.9999)
    
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    model_ema.ema.load_state_dict(checkpoint['model_ema'])
    model_ema.ema.eval()
    
    preds = []
    with torch.no_grad():
        for images, _ in data_loader_val:
            images = images.cuda()
            out = model_ema.ema(images)
            preds.extend(out.argmax(dim=-1).cpu().tolist())
    return preds

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
    
    config.defrost()
    config.MODEL.NUM_CLASSES = 5
    config.freeze()
    
    _, dataset_val, _, _, data_loader_val, _, _ = build_loader(config)
    
    print("Getting predictions for best_ckpt_ema.pth...")
    preds_best = get_predictions("output/tiny/finetune/best_ckpt_ema.pth", config, data_loader_val)
    
    print("Getting predictions for latest_ckpt.pth...")
    preds_latest = get_predictions("output/tiny/finetune/latest_ckpt.pth", config, data_loader_val)
    
    assert len(preds_best) == len(preds_latest), f"Length mismatch: {len(preds_best)} vs {len(preds_latest)}"
    
    identical_count = sum(1 for p1, p2 in zip(preds_best, preds_latest) if p1 == p2)
    different_count = len(preds_best) - identical_count
    
    print(f"\nTotal validation samples: {len(preds_best)}")
    print(f"Number of samples with identical predictions: {identical_count}")
    print(f"Number of samples with different predictions: {different_count}")
    
    # Print targets to check labels too
    targets = []
    for _, target in data_loader_val:
        targets.extend(target.tolist())
        
    correct_best = sum(1 for p, t in zip(preds_best, targets) if p == t)
    correct_latest = sum(1 for p, t in zip(preds_latest, targets) if p == t)
    
    print(f"Best EMA correct:   {correct_best} ({100.0 * correct_best / len(targets):.3f}%)")
    print(f"Latest EMA correct: {correct_latest} ({100.0 * correct_latest / len(targets):.3f}%)")

if __name__ == '__main__':
    main()
