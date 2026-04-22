import json, sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

with open(r'c:\Users\chira\Downloads\idp\fall_detection_training (2).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cells with feature extraction or binary/2-class hints
keywords = ['num_features', 'NUM_FEATURES', '20', 'binary', 'BINARY', 'label', 'CLASS', 'class_map',
            'feat', 'extract', 'feature', 'def extract', 'num_classes', 'NUM_CLASSES']
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(kw in src for kw in keywords) and ('def ' in src or 'NUM_FEATURES' in src or 'num_classes' in src or 'binary' in src.lower() or 'label' in src):
        print(f'--- Cell {i} ---')
        print(src[:2000])
        print()
