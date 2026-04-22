import sys
sys.path.insert(0, r'c:\Users\chira\Downloads\idp\hf_space')

import torch
from model import TransformerCNNLSTM

ckpt = torch.load(r'c:\Users\chira\Downloads\idp\best_v5.pth', map_location='cpu', weights_only=False)
sd = ckpt['sd']

model = TransformerCNNLSTM()
result = model.load_state_dict(sd, strict=True)
print("Missing:", result.missing_keys)
print("Unexpected:", result.unexpected_keys)
model.eval()

# dummy forward pass
x = torch.randn(1, 40, 20)  # (batch, window, features)
with torch.no_grad():
    probs = model.predict_proba(x)
print("Output shape:", probs.shape)
print("Probs sum:", probs.sum().item())
print("\n✓ Model loads and runs correctly!")
print(f"  Classes: [fall={probs[0][0]:.3f}, no_fall={probs[0][1]:.3f}]")
