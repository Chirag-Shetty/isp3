import json, sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

with open(r'c:\Users\chira\Downloads\idp\fall_detection_training (2).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find feature extraction function
keywords = ['def extract', 'pointCloud', 'trackData', 'heightData', 'NUM_FEATURES', 'trackData', 'feat', 'append']
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'def extract' in src or ('pointCloud' in src and 'feature' in src.lower()):
        print(f'--- Cell {i} ---')
        print(src[:3000])
        print()
