import torch

def main():
    latest = torch.load("output/tiny/finetune/latest_ckpt.pth", map_location='cpu')
    best_ema = torch.load("output/tiny/finetune/best_ckpt_ema.pth", map_location='cpu')
    
    model_latest = latest['model']
    model_best = best_ema['model']
    
    key = 'stem.0.weight'
    w_latest = model_latest[key]
    w_best = model_best[key]
    
    print(f"Student {key}:")
    print(f"Latest mean: {w_latest.mean().item():.8f}, std: {w_latest.std().item():.8f}")
    print(f"Best mean:   {w_best.mean().item():.8f}, std:   {w_best.std().item():.8f}")
    print(f"Max absolute change in student backbone: {(w_latest - w_best).abs().max().item():.8f}")

if __name__ == '__main__':
    main()
