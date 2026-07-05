"""
Diagnostic script: Inspect EMA state inside a checkpoint.

Compares student model weights vs EMA weights to determine whether:
1. They are identical (EMA never updated)
2. They are different (EMA is updating)
3. EMA has the wrong number of classes (head mismatch)
4. EMA logits are degenerate (all zeros, all same, etc.)

Also performs a forward pass with both student and EMA to compare outputs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from config import get_config
from models import build_model
from timm.utils import ModelEma
from copy import deepcopy


def main():
    # Load the latest checkpoint
    ckpt_path = os.path.join('output', 'tiny', 'finetune', 'latest_ckpt.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join('output', 'tiny', 'finetune', 'best_ckpt.pth')
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    print(f"\n{'='*70}")
    print("CHECKPOINT CONTENTS")
    print(f"{'='*70}")
    print(f"Keys: {list(checkpoint.keys())}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"max_accuracy: {checkpoint.get('max_accuracy', 'N/A')}")
    print(f"max_accuracy_ema: {checkpoint.get('max_accuracy_ema', 'N/A')}")
    print(f"max_accuray_ema (typo key): {checkpoint.get('max_accuray_ema', 'N/A')}")
    
    has_model = 'model' in checkpoint
    has_ema = 'model_ema' in checkpoint
    print(f"\nHas 'model': {has_model}")
    print(f"Has 'model_ema': {has_ema}")
    
    if not has_model or not has_ema:
        print("CRITICAL: Missing model or model_ema in checkpoint!")
        return
    
    model_sd = checkpoint['model']
    ema_sd = checkpoint['model_ema']
    
    # ---------------------------------------------------------------
    # 1. Compare keys
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("KEY COMPARISON")
    print(f"{'='*70}")
    model_keys = set(model_sd.keys())
    ema_keys = set(ema_sd.keys())
    print(f"Student key count: {len(model_keys)}")
    print(f"EMA key count: {len(ema_keys)}")
    
    only_in_model = model_keys - ema_keys
    only_in_ema = ema_keys - model_keys
    if only_in_model:
        print(f"\nKeys ONLY in student (first 10): {list(only_in_model)[:10]}")
    if only_in_ema:
        print(f"\nKeys ONLY in EMA (first 10): {list(only_in_ema)[:10]}")
    if not only_in_model and not only_in_ema:
        print("All keys match between student and EMA.")
    
    # ---------------------------------------------------------------
    # 2. Compare head (classifier) weights
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("HEAD / CLASSIFIER COMPARISON")
    print(f"{'='*70}")
    
    head_keys = [k for k in model_keys if 'head' in k.lower() or 'classifier' in k.lower() or 'fc.' in k.lower()]
    print(f"Head-related keys: {head_keys}")
    
    for key in head_keys:
        if key in ema_sd:
            m_tensor = model_sd[key]
            e_tensor = ema_sd[key]
            print(f"\n  {key}:")
            print(f"    Student shape: {m_tensor.shape}, dtype: {m_tensor.dtype}")
            print(f"    EMA shape:     {e_tensor.shape}, dtype: {e_tensor.dtype}")
            
            if m_tensor.shape != e_tensor.shape:
                print(f"    *** SHAPE MISMATCH! ***")
            else:
                diff = (m_tensor.float() - e_tensor.float()).abs()
                print(f"    Max diff: {diff.max().item():.8f}")
                print(f"    Mean diff: {diff.mean().item():.8f}")
                print(f"    Are identical: {torch.equal(m_tensor, e_tensor)}")
                
                # Print actual values for small tensors
                if m_tensor.numel() <= 20:
                    print(f"    Student values: {m_tensor.flatten().tolist()}")
                    print(f"    EMA values:     {e_tensor.flatten().tolist()}")
                else:
                    print(f"    Student first 5: {m_tensor.flatten()[:5].tolist()}")
                    print(f"    EMA first 5:     {e_tensor.flatten()[:5].tolist()}")
                    print(f"    Student mean: {m_tensor.float().mean().item():.8f}, std: {m_tensor.float().std().item():.8f}")
                    print(f"    EMA mean:     {e_tensor.float().mean().item():.8f}, std: {e_tensor.float().std().item():.8f}")
        else:
            print(f"  {key}: NOT FOUND in EMA!")
    
    # ---------------------------------------------------------------
    # 3. Compare ALL parameter norms
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PARAMETER NORM COMPARISON (first 10 + last 5)")
    print(f"{'='*70}")
    
    common_keys = sorted(model_keys & ema_keys)
    identical_count = 0
    different_count = 0
    total_max_diff = 0.0
    
    for i, key in enumerate(common_keys):
        m = model_sd[key].float()
        e = ema_sd[key].float()
        if m.shape != e.shape:
            print(f"  {key}: SHAPE MISMATCH student={m.shape} ema={e.shape}")
            continue
        diff = (m - e).abs()
        max_diff = diff.max().item()
        total_max_diff = max(total_max_diff, max_diff)
        is_identical = torch.equal(model_sd[key], ema_sd[key])
        if is_identical:
            identical_count += 1
        else:
            different_count += 1
        
        if i < 10 or i >= len(common_keys) - 5:
            print(f"  {key}: norm_s={m.norm().item():.4f} norm_e={e.norm().item():.4f} "
                  f"max_diff={max_diff:.8f} identical={is_identical}")
    
    print(f"\nSummary: {identical_count} identical, {different_count} different out of {len(common_keys)} total")
    print(f"Overall max diff: {total_max_diff:.8f}")
    
    if identical_count == len(common_keys):
        print("\n*** CRITICAL: ALL PARAMETERS ARE IDENTICAL! EMA IS NEVER UPDATING! ***")
    elif different_count == len(common_keys):
        print("\n  All parameters differ (expected for a trained EMA).")
    
    # ---------------------------------------------------------------
    # 4. Forward pass comparison
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FORWARD PASS COMPARISON")
    print(f"{'='*70}")
    
    # Build model
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cfg', type=str, default='classification/configs/DAMamba/damamba_tiny.yaml')
    args_fake, _ = parser.parse_known_args([])
    
    # Minimal config
    from yacs.config import CfgNode
    
    # Just build the model with the right num_classes
    model = build_model_from_sd(model_sd)
    if model is None:
        print("Could not build model for forward pass test.")
        return
    
    model.eval()
    model.cuda()
    
    ema_model = deepcopy(model)
    ema_model.eval()
    
    # Load weights
    msg_s = model.load_state_dict(model_sd, strict=False)
    msg_e = ema_model.load_state_dict(ema_sd, strict=False)
    print(f"Student load: {msg_s}")
    print(f"EMA load: {msg_e}")
    
    # Random input
    x = torch.randn(2, 3, 224, 224).cuda()
    
    with torch.no_grad():
        out_s = model(x)
        out_e = ema_model(x)
    
    print(f"\nStudent output shape: {out_s.shape}")
    print(f"EMA output shape: {out_e.shape}")
    print(f"Student logits [0]: {out_s[0].cpu().tolist()}")
    print(f"EMA logits [0]:     {out_e[0].cpu().tolist()}")
    print(f"Student softmax [0]: {out_s[0].softmax(dim=-1).cpu().tolist()}")
    print(f"EMA softmax [0]:     {out_e[0].softmax(dim=-1).cpu().tolist()}")
    print(f"Student argmax: {out_s.argmax(dim=-1).cpu().tolist()}")
    print(f"EMA argmax:     {out_e.argmax(dim=-1).cpu().tolist()}")
    
    # Check for degenerate EMA outputs
    ema_probs = out_e.softmax(dim=-1)
    print(f"\nEMA prob stats: min={ema_probs.min().item():.6f}, max={ema_probs.max().item():.6f}, "
          f"mean={ema_probs.mean().item():.6f}, std={ema_probs.std().item():.6f}")
    
    if ema_probs.std().item() < 1e-6:
        print("*** CRITICAL: EMA outputs are essentially constant (degenerate)! ***")


def build_model_from_sd(state_dict):
    """Build a DAMamba model that matches the state dict."""
    try:
        # Infer num_classes from head weight shape
        for key in state_dict:
            if 'head' in key and 'weight' in key:
                num_classes = state_dict[key].shape[0]
                print(f"Inferred num_classes={num_classes} from {key} shape={state_dict[key].shape}")
                break
        else:
            print("Could not find head weight to infer num_classes")
            return None
        
        # Import and build
        from models.DAMamba import DAMamba
        model = DAMamba(
            img_size=224,
            in_chans=3,
            num_classes=num_classes,
            depths=[3, 4, 12, 5],
            dims=[80, 160, 320, 512],
            head_dim=16,
            mlp_ratios=[4, 4, 3, 3],
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
