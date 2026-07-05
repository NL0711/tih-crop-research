import torch

def main():
    ckpt_path = "output/tiny/finetune/latest_ckpt.pth"
    print(f"Loading checkpoint {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    print("\nKeys in checkpoint:")
    print(checkpoint.keys())
    
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"max_accuracy: {checkpoint.get('max_accuracy')}")
    print(f"max_accuracy_ema: {checkpoint.get('max_accuracy_ema')}")
    print(f"max_accuray_ema: {checkpoint.get('max_accuray_ema')}")
    
    model_state = checkpoint['model']
    ema_state = checkpoint['model_ema']
    
    # Check head FC weight
    student_head_w = model_state['head.fc.weight']
    ema_head_w = ema_state['head.fc.weight']
    
    print(f"\nStudent head fc weight shape: {student_head_w.shape}")
    print(f"EMA head fc weight shape: {ema_head_w.shape}")
    
    diff = (student_head_w.float() - ema_head_w.float()).abs()
    print(f"Head weight max diff: {diff.max().item()}")
    print(f"Head weight mean diff: {diff.mean().item()}")
    print(f"Are head weights equal? {torch.equal(student_head_w, ema_head_w)}")
    
    # Check other layers
    print("\nComparing all common layers:")
    identical_count = 0
    different_count = 0
    total_keys = 0
    
    for k in model_state.keys():
        if k in ema_state:
            total_keys += 1
            if torch.equal(model_state[k], ema_state[k]):
                identical_count += 1
            else:
                different_count += 1
                
    print(f"Total compared keys: {total_keys}")
    print(f"Identical keys: {identical_count}")
    print(f"Different keys: {different_count}")

if __name__ == '__main__':
    main()
