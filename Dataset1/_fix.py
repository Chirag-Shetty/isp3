from pathlib import Path
p = Path('evaluate.py')
c = p.read_text(encoding='utf-8')

# remove any duplicate _HERE / DATASET_ROOT lines added by previous attempts
lines = c.split('\n')
lines = [l for l in lines if not l.startswith('_HERE') and not l.startswith('DATASET_ROOT')]
c = '\n'.join(lines)

# insert after "from pathlib import Path"
marker = 'from pathlib import Path'
idx = c.find(marker)
if idx == -1:
    print('ERROR: marker not found')
else:
    insert_pos = idx + len(marker)
    insert_text = '\n\n_HERE        = Path(__file__).resolve().parent\nDATASET_ROOT = _HERE / "Dataset1"  # fall_stand/, sitting_chair/ etc.'
    c = c[:insert_pos] + insert_text + c[insert_pos:]
    p.write_text(c, encoding='utf-8')
    print('Fix applied successfully')
