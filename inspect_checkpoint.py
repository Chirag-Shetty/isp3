import torch

checkpoint = torch.load(
    r'c:\Users\chira\Downloads\idp\best_v5.pth',
    map_location='cpu',
    weights_only=False
)

print("Type:", type(checkpoint))
if isinstance(checkpoint, dict):
    print("Keys:", list(checkpoint.keys()))
else:
    print("Not a dict — it's a raw state_dict or model object")
