"""CLI prior prediction accepts design inputs only, not observed y."""
import json
from pathlib import Path
p = Path('design.json')
with p.open('x') as f:
    data = json.loads(Path('input/data.json').read_text())
    json.dump({k: data[k] for k in ('x', 'clinic_design')}, f)
