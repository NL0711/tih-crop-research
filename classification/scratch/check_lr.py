import torch

def main():
    latest = torch.load("output/tiny/finetune/latest_ckpt.pth", map_location='cpu')
    
    print("Optimizer state keys:")
    print(latest['optimizer'].keys())
    
    # Check learning rate in param_groups
    for idx, group in enumerate(latest['optimizer']['param_groups']):
        print(f"Param group {idx} keys: {group.keys()}")
        print(f"lr: {group.get('lr')}")
        print(f"initial_lr: {group.get('initial_lr')}")

if __name__ == '__main__':
    main()
