from pathlib import Path

p = Path('evaluate.py')
c = p.read_text(encoding='utf-8')

# Replace all unicode arrows
c = c.replace('\u2192', '->')

# Fix the config block — ensure _HERE and DATASET_ROOT are defined correctly
old_block = '_HERE         = __file__\nDATASET_ROOT  = _HERE\nMODEL_PATH    = _HERE'
new_block = '_HERE         = Path(__file__).resolve().parent\nDATASET_ROOT  = _HERE   # fall_stand/, sitting_chair/ etc. live here\nMODEL_PATH    = _HERE'
c = c.replace(old_block, new_block)

p.write_text(c, encoding='utf-8')
print('Patched OK')
print('DATASET_ROOT check:', 'DATASET_ROOT  = _HERE' in c)
print('_HERE check:', 'Path(__file__).resolve().parent' in c)
