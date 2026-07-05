import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config import get_config
from models import build_model
from data import build_loader
from utils.utils import reduce_tensor

def validate_model(data_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.cuda()
            targets = targets.cuda()
            outputs = model(images)
            preds = outputs.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total * 100.0

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
    
    print("Evaluating models on validation set...")
    
    # 1. Student accuracy
    acc_student = validate_model(data_loader_val, model)
    print(f"1. Student accuracy: {acc_student:.3f}%")
    
    # 2. Original EMA accuracy
    acc_ema_orig = validate_model(data_loader_val, model_ema.ema)
    print(f"2. Original EMA accuracy: {acc_ema_orig:.3f}%")
    
    # 3. Copy student head only
    print("\nCopying student head to EMA model...")
    # Save original EMA head
    orig_ema_head_state = {k: v.clone() for k, v in model_ema.ema.head.state_dict().items()}
    # Load student head
    model_ema.ema.head.load_state_dict(model.head.state_dict())
    acc_ema_copy_head = validate_model(data_loader_val, model_ema.ema)
    print(f"3. EMA accuracy with copied student head: {acc_ema_copy_head:.3f}%")
    
    # 4. Copy student backbone parameters, keep EMA buffers (like BN stats)
    print("\nRestoring EMA head. Copying student backbone parameters to EMA (but keeping EMA buffers)...")
    model_ema.ema.head.load_state_dict(orig_ema_head_state)
    
    # Copy parameters only
    for (name_e, p_e), (name_s, p_s) in zip(model_ema.ema.named_parameters(), model.named_parameters()):
        p_e.data.copy_(p_s.data)
        
    acc_ema_copy_params = validate_model(data_loader_val, model_ema.ema)
    print(f"4. EMA accuracy with copied student parameters (EMA buffers): {acc_ema_copy_params:.3f}%")
    
    # 5. Copy student buffers to EMA (now EMA is identical to student)
    print("\nCopying student buffers to EMA...")
    for (name_e, b_e), (name_s, b_s) in zip(model_ema.ema.named_buffers(), model.named_buffers()):
        b_e.copy_(b_s)
        
    acc_ema_copy_all = validate_model(data_loader_val, model_ema.ema)
    print(f"5. EMA accuracy with copied student parameters and buffers: {acc_ema_copy_all:.3f}%")

if __name__ == '__main__':
    main()
