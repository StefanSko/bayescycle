"""Initial-stage integrity checks and exact-byte provenance, no refitting."""
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import os
import platform
import numpy as np
import jax


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert jax.config.x64_enabled
assert len(jax.devices('cpu')) == 4
for stage, name in [('initial','data'), ('recovery','recovery')]:
    run = Path('results')/stage
    source = Path('input')/(name+'.json')
    raw = json.loads(source.read_bytes())
    canonical = json.loads((run/'data.json').read_bytes())
    assert canonical['format'] == 'bayescycle.data.json.v1'
    assert set(raw) == set(canonical['variables']) == {'x','y','clinic_design'}
    for k,v in canonical['variables'].items():
        a = np.asarray(v['values']).reshape(v['shape'])
        np.testing.assert_array_equal(a, np.asarray(raw[k]))
        assert v['dtype'] == 'float64'
    assert np.asarray(raw['clinic_design']).shape == (160,8)
    np.testing.assert_array_equal(np.sum(raw['clinic_design'], axis=0), np.full(8,20))
    np.testing.assert_array_equal(np.sum(raw['clinic_design'], axis=1), np.ones(160))
    assert set(np.unique(raw['clinic_design'])) == {0.,1.}
    native = json.loads((run/'run.json').read_bytes())
    assert native['backend'] == 'bayesjax'
    assert native['model']['sha256'] == 'sha256:'+sha(Path('model.py'))
    assert native['inputs'][0]['sha256'] == 'sha256:'+sha(source)
assert (Path('results/initial/model.ir.json').read_bytes() == Path('results/recovery/model.ir.json').read_bytes() == Path('results/prior/model.ir.json').read_bytes())
paths = [Path('model.py'),Path('analysis.py'),Path('prepare_design.py'),Path('run_initial.sh'),Path('record_provenance.py'),Path('design.json'),*Path('input').glob('*.json')]
paths += [p for stage in ['initial','recovery','prior'] for p in (Path('results')/stage).rglob('*') if p.is_file()]
record = {'stage':'initial only; awaits human scientific review', 'sha256':{str(p):sha(p) for p in sorted(paths)},
    'versions': {k:md.version(k) for k in ['bayeswire','bayesjax','bayescycle','jax','jaxlib','blackjax','arviz','numpy','matplotlib']},
    'python':platform.python_version(), 'devices':[str(d) for d in jax.devices()],
    'environment':{k:os.environ.get(k) for k in ['JAX_ENABLE_X64','XLA_FLAGS','MPLBACKEND','OMP_NUM_THREADS']},
    'checks':['native source hashes match original bytes','canonical fit arrays equal original input arrays','float64 data; 4 CPU devices; x64 enabled','identical initial/recovery/prior model IR','8 one-hot clinics, 20 observations each'],
    'notes':'NDJSON fingerprint/header/trailer and draw uniqueness checked by analysis.py. Native diagnostics retained in posterior.ndjson trailers; distinct from ArviZ rank diagnostics.'}
with Path('provenance.json').open('x') as f:
    json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k!='sha256'},indent=2))
