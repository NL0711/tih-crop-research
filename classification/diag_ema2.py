import torch
print("PyTorch:", torch.__version__)
import torch.nn as nn

m = nn.Linear(512, 5)
sd = {'weight': torch.randn(1000, 512), 'bias': torch.randn(1000)}
try:
    r = m.load_state_dict(sd, strict=False)
    print('Loaded! Shape:', m.weight.shape)
    print('Result:', r)
except Exception as e:
    print('Error:', type(e).__name__, e)

# Now test what the actual checkpoint loading does:
# First: the model is built with 5 classes
# Then: load the RESUME checkpoint which has 1000 classes
# The resume log says: resuming model: <All keys matched successfully>
# This means the checkpoint actually has MATCHING keys. So the checkpoint
# was saved from a model that ALREADY had 1000 classes loaded.

# Let me check the actual saved checkpoint
ckpt = torch.load('output/tiny/finetune/latest_ckpt.pth', map_location='cpu')
model_sd = ckpt['model']
ema_sd = ckpt['model_ema']

# Check number of output classes in both
for key in ['head.fc.weight', 'head.fc.bias']:
    print(f"\ncheckpoint model[{key}]: {model_sd[key].shape}")
    print(f"checkpoint ema[{key}]:   {ema_sd[key].shape}")

# Now build a fresh model with 5 classes and try to load this
from models.DAMamba import DAMamba
model_5 = DAMamba(
    img_size=224, in_chans=3, num_classes=5,
    depths=[3, 4, 12, 5], dims=[80, 160, 320, 512],
    head_dim=16, mlp_ratios=[4, 4, 3, 3],
    drop_rate=0.0, drop_path_rate=0.0,
)
print(f"\nFresh model head: {model_5.head.fc.weight.shape}")

try:
    msg = model_5.load_state_dict(model_sd, strict=False)
    print(f"Load result: {msg}")
    print(f"After load head: {model_5.head.fc.weight.shape}")
except RuntimeError as e:
    err_str = str(e)
    # Only print first 500 chars
    print(f"RuntimeError: {err_str[:500]}")
