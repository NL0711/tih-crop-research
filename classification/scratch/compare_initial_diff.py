import torch

def main():
    best_ema = torch.load("output/tiny/finetune/best_ckpt_ema.pth", map_location='cpu')
    
    model = best_ema['model']
    ema = best_ema['model_ema']
    
    key = 'stem.0.weight'
    w_student = model[key]
    w_ema = ema[key]
    
    print(f"Initial {key} difference at Epoch 0:")
    print(f"Student mean: {w_student.mean().item():.8f}, std: {w_student.std().item():.8f}")
    print(f"EMA mean:     {w_ema.mean().item():.8f}, std:     {w_ema.std().item():.8f}")
    print(f"Max absolute diff: {(w_student - w_ema).abs().max().item():.8f}")
    print(f"Mean absolute diff: {(w_student - w_ema).abs().mean().item():.8f}")

if __name__ == '__main__':
    main()
