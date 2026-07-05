import torch

def main():
    latest = torch.load("output/tiny/finetune/latest_ckpt.pth", map_location='cpu')
    
    model = latest['model']
    ema = latest['model_ema']
    
    # Let's compare stem.0.weight
    key = 'stem.0.weight'
    w_student = model[key]
    w_ema = ema[key]
    
    print(f"Comparing {key}:")
    print(f"Student mean: {w_student.mean().item():.8f}, std: {w_student.std().item():.8f}")
    print(f"EMA mean:     {w_ema.mean().item():.8f}, std:     {w_ema.std().item():.8f}")
    print(f"Max absolute diff: {(w_student - w_ema).abs().max().item():.8f}")
    print(f"Mean absolute diff: {(w_student - w_ema).abs().mean().item():.8f}")
    
    # Compare another weight in block 0
    key = 'layers.0.blocks.0.mixer.A_log'
    if key in model:
        w_student = model[key]
        w_ema = ema[key]
        print(f"\nComparing {key}:")
        print(f"Student mean: {w_student.mean().item():.8f}, std: {w_student.std().item():.8f}")
        print(f"EMA mean:     {w_ema.mean().item():.8f}, std:     {w_ema.std().item():.8f}")
        print(f"Max absolute diff: {(w_student - w_ema).abs().max().item():.8f}")

if __name__ == '__main__':
    main()
