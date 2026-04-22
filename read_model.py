import json, sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

with open(r'c:\Users\chira\Downloads\idp\fall_detection_training (2).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cells with model class definition
keywords = ['class ', 'TransformerCNN', 'nn.Module', 'def forward', 'input_proj', 'proj', 'PositionalEncoding', 'AttentionPool']
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(kw in src for kw in keywords):
        print(f'--- Cell {i} ({cell["cell_type"]}) ---')
        print(src[:3000])
        print()
