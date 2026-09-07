"""Read-only artifact checks, without sampling or overwriting any output."""
from pathlib import Path
import hashlib
import json
import numpy as np
import arviz as az

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'results/provenance.json').read_text())
for name, expected in manifest['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('All current provenance hashes verified.')
for stage in ['recovery', 'initial']:
    folder = root / 'results' / stage
    with np.load(folder / 'posterior.npz') as archive:
        draws = dict(archive)
    assert set(draws) == {'alpha','beta','tau','z','sigma'}
    for k,v in draws.items():
        assert v.shape == ((4,1000,8) if k == 'z' else (4,1000)), (k,v.shape)
        assert v.dtype == np.float64 and np.isfinite(v).all()
    b = draws['beta'].ravel()
    summary = az.summary(az.from_dict(posterior=draws), round_to='none')
    got = dict(mean=float(b.mean()), sd=float(b.std(ddof=1)), q025=float(np.quantile(b,.025)), q975=float(np.quantile(b,.975)))
    got.update({k:float(summary.loc['beta', col]) for k,col in [('mcse_mean','mcse_mean'),('rhat','r_hat'),('ess_bulk','ess_bulk'),('ess_tail','ess_tail')]})
    saved = json.loads((folder / 'result.json').read_text())
    for k,v in got.items():
        assert np.isclose(v,saved['beta'][k],rtol=1e-12,atol=1e-12), k
    with np.load(folder / 'native_diagnostics.npz') as native:
        assert int(native['sampling_is_divergent'].sum()) == saved['divergences']
    print(stage, json.dumps(got))
with np.load(root / 'scratch-export-failure/results/recovery/posterior.npz') as old, np.load(root / 'results/recovery/posterior.npz') as new:
    identical = all(np.array_equal(old[k],new[k]) for k in new.files)
print('Recovery repair retry draws exactly equal to saved failed-export attempt:', identical)
assert identical
