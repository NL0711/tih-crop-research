import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config import get_config
from models import build_model
from data import build_loader

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
    
    model = build_model(config)
    model.cuda()
    from timm.utils import ModelEma
    model_ema = ModelEma(model, decay=0.9999)
    
    checkpoint = torch.load("output/tiny/finetune/latest_ckpt.pth", map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    model_ema.ema.load_state_dict(checkpoint['model_ema'])
    
    model.eval()
    model_ema.ema.eval()
    
    # We want to hook the input to the head.fc
    features_s = []
    features_e = []
    
    def hook_s(module, input, output):
        features_s.append(input[0].cpu())
        
    def hook_e(module, input, output):
        features_e.append(input[0].cpu())
        
    model.head.fc.register_forward_hook(hook_s)
    model_ema.ema.head.fc.register_forward_hook(hook_e)
    
    with torch.no_grad():
        for images, _ in data_loader_val:
            images = images.cuda()
            _ = model(images)
            _ = model_ema.ema(images)
            break # Just one batch is enough
            
    feat_s = features_s[0]
    feat_e = features_e[0]
    
    print("Backbone features (before FC):")
    print(f"Student: mean={feat_s.mean().item():.4f}, std={feat_s.std().item():.4f}")
    print(f"EMA:     mean={feat_e.mean().item():.4f}, std={feat_e.std().item():.4f}")
    
    # Let's also print the head weights
    w_s = model.head.fc.weight.cpu()
    w_e = model_ema.ema.head.fc.weight.cpu()
    
    print("\nHead FC weights:")
    print(f"Student: mean={w_s.mean().item():.6f}, std={w_s.std().item():.6f}")
    print(f"EMA:     mean={w_e.mean().item():.6f}, std={w_e.std().item():.6f}")

if __name__ == '__main__':
    main()
