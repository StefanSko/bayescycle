"""Read-only artifact checks, without sampling or overwriting any output."""
from pathlib import Path
import hashlib
import json
import numpy as np
import arviz as az

root = Path(__file__).resolve().parent
preserved = json.loads((root / 'revision-preservation.json').read_text())
for name, expected in preserved.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('All', len(preserved), 'preserved baseline files unchanged.')
initial_model = (root / 'model.py').read_text()
revised_model = (root / 'model_revised.py').read_text()
assert revised_model == initial_model.replace('Fixed initial evaluator fixture; not scientifically approved.', 'Scripted revised evaluator fixture; awaits human scientific review.').replace('beta = Param(Normal(0.0, 1.0))', 'beta = Param(Normal(0.0, 0.25))')
manifest = json.loads((root / 'results/revised-provenance.json').read_text())
for name, expected in manifest['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('All current provenance hashes verified.')
for stage in ['revised-recovery', 'revised']:
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
        assert int(native['warmup_is_divergent'].sum()) == saved['warmup_divergences']
    assert saved['sampler'] == dict(backend='bayesjax / BlackJAX NUTS', seed=4301 if stage == 'revised-recovery' else 4201, dtype='float64', num_chains=4, num_warmup=500, num_samples=1000, target_acceptance_rate=0.9, max_tree_depth=10)
    for path in [saved['paths']['samples'], saved['paths']['native_diagnostics'], saved['paths']['prior_predictive'], *saved['paths']['figures']]:
        assert (root / path).is_file(), path
    print(stage, json.dumps(got))
