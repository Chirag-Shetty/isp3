import torch

ckpt = torch.load(r'c:\Users\chira\Downloads\idp\best_v5.pth', map_location='cpu', weights_only=False)
sd = ckpt['sd']

print("=== All keys and shapes ===")
for k, v in sd.items():
    if hasattr(v, 'shape'):
        print(f"  {k:55s} {list(v.shape)}")
    else:
        print(f"  {k:55s} (scalar: {v})")
