import torch

def main():
    checkpoint = torch.load('classification/weights/DAMamba-T.pth', map_location='cpu')
    print("Pretrained checkpoint keys:")
    print(checkpoint.keys())
    
    if 'model' in checkpoint:
        print("model keys count:", len(checkpoint['model'].keys()))
    if 'model_ema' in checkpoint:
        print("model_ema keys count:", len(checkpoint['model_ema'].keys()))
    else:
        print("model_ema NOT in checkpoint!")

if __name__ == '__main__':
    main()
