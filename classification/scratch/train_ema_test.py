import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from config import get_config
from models import build_model
from data import build_loader
from utils.utils import NativeScalerWithGradNormCount, load_pretrained_ema
from timm.utils import ModelEmaV3

class ModelEmaV3Wrapper(ModelEmaV3):
    @property
    def ema(self):
        return self.module

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
    
    # Create loader
    dataset_train, dataset_val, _, data_loader_train, data_loader_val, _, _ = build_loader(config)
    
    # Create model
    model = build_model(config)
    model.cuda()
    
    # Initialize ModelEmaV3Wrapper with warmup and excluding buffers
    model_ema = ModelEmaV3Wrapper(
        model,
        decay=0.9999,
        use_warmup=True,
        exclude_buffers=False,
        device=torch.device('cuda')
    )
    
    # Load pretrained weights (filtered)
    import logging
    logger = logging.getLogger("dummy")
    load_pretrained_ema(config, model, logger, model_ema)
    
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_scaler = NativeScalerWithGradNormCount()
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Starting fast training for 3 epochs with ModelEmaV3Wrapper...")
    for epoch in range(3):
        model.train()
        for idx, (samples, targets) in enumerate(data_loader_train):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(samples)
                loss = criterion(outputs, targets)
                
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, parameters=model.parameters())
            
            # Update EMA with step count
            global_step = epoch * len(data_loader_train) + idx
            model_ema.update(model, step=global_step)
            
        # Validate
        acc_s = validate_model(data_loader_val, model)
        acc_e = validate_model(data_loader_val, model_ema.ema)
        decay_val = model_ema.get_decay(epoch * len(data_loader_train))
        print(f"Epoch {epoch}: Student Acc = {acc_s:.3f}%, EMA Acc = {acc_e:.3f}%, Current EMA decay = {decay_val:.6f}")

if __name__ == '__main__':
    main()
