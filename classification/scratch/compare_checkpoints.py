import torch

def main():
    latest = torch.load("output/tiny/finetune/latest_ckpt.pth", map_location='cpu')
    best_ema = torch.load("output/tiny/finetune/best_ckpt_ema.pth", map_location='cpu')
    
    print(f"Latest epoch: {latest.get('epoch')}")
    print(f"Best EMA epoch: {best_ema.get('epoch')}")
    
    latest_ema_sd = latest['model_ema']
    best_ema_sd = best_ema['model_ema']
    
    # Compare some weights
    key = 'head.fc.weight'
    w_latest = latest_ema_sd[key]
    w_best = best_ema_sd[key]
    
    print(f"\nComparing {key}:")
    print(f"Latest EMA weight mean: {w_latest.mean().item():.8f}, std: {w_latest.std().item():.8f}")
    print(f"Best EMA weight mean: {w_best.mean().item():.8f}, std: {w_best.std().item():.8f}")
    print(f"Are they identical? {torch.equal(w_latest, w_best)}")
    print(f"Max absolute difference: {(w_latest - w_best).abs().max().item():.8f}")
    
    # Compare all keys
    diff_keys = []
    for k in latest_ema_sd.keys():
        if not torch.equal(latest_ema_sd[k], best_ema_sd[k]):
            diff_keys.append(k)
            
    print(f"\nOut of {len(latest_ema_sd)} total keys, {len(diff_keys)} keys differ between latest and best_ema checkpoints.")
    if diff_keys:
        print(f"First 10 differing keys: {diff_keys[:10]}")

if __name__ == '__main__':
    main()
