"""Verify revision preservation/provenance and reconstruct beta without refitting.
uv run --project .. python verify_revision.py
Add --record once to exclusively create revision-provenance.json.
"""
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import sys
import arviz as az
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

snapshot = json.loads(Path('preservation-before-revision.json').read_text())
for path, digest in snapshot.items():
    assert sha(path) == digest, path
initial = Path('model.py').read_text()
expected = initial.replace('Fixed initial engineering fixture', 'Scripted revised proposal').replace('beta = Param(Normal(0.0, 1.0))', 'beta = Param(Normal(0.0, 0.25))')
assert Path('model_revised.py').read_text() == expected
ir = json.loads(Path('results/initial/model.ir.json').read_text())
for key in ['params', 'stochastic_sites']:
    beta = next(p for p in ir['model'][key] if p['name'] == 'beta')
    distribution = beta['value']['distribution'] if key == 'params' else beta['distribution']
    assert distribution['scale']['value'] == 1.0
    distribution['scale']['value'] = 0.25
summaries = {}
for stage, old, source, seed in [('revised', 'initial', 'input/data.json', 4201), ('revised-recovery', 'recovery', 'input/recovery.json', 4301), ('revised-prior', 'prior', 'design.json', 4200)]:
    run = Path('results') / stage
    assert json.loads((run/'model.ir.json').read_text()) == ir
    assert (run/'data.json').read_bytes() == (Path('results')/old/'data.json').read_bytes()
    native = json.loads((run/'run.json').read_text())
    assert native['backend'] == 'bayesjax'
    assert native['model']['sha256'] == 'sha256:' + sha('model_revised.py')
    assert native['inputs'][0]['sha256'] == 'sha256:' + sha(source)
    assert native['settings'] == json.loads((Path('results')/old/'run.json').read_text())['settings']
    if stage == 'revised-prior':
        with np.load(run/'prior_predictive.npz') as p:
            assert p['y'].shape == (200,160) and np.isfinite(p['y']).all()
        continue
    result = json.loads((run/'result.json').read_text())
    baseline = json.loads((Path('results')/old/'result.json').read_text())
    assert set(result) == set(baseline), 'Result schema keys differ'
    assert result['sampler'] == baseline['sampler']
    for path, digest in result['sha256'].items():
        assert sha(path) == digest
    rows = [json.loads(line) for line in (run/'posterior.ndjson').read_text().splitlines()]
    fingerprint = 'sha256:' + hashlib.sha256(b'bayescycle-model-data-v1\n' + (run/'model.ir.json').read_bytes() + b'\n' + (run/'data.json').read_bytes()).hexdigest()
    assert fingerprint == result['model_data_fingerprint'] == rows[0]['model_data_fingerprint'] == rows[-1]['trailer']['model_data_fingerprint']
    with np.load(run/'posterior.npz') as p:
        samples = dict(p)
    for key, a in samples.items():
        assert a.shape == ((4,1000,8) if key == 'z' else (4,1000))
        assert a.dtype == np.float64 and np.isfinite(a).all()
    for row in rows[1:-1]:
        for key, a in samples.items():
            np.testing.assert_array_equal(a[row['chain'],row['draw']], row['values'][key])
    b = samples['beta']
    idata = az.from_dict(posterior={'beta':b})
    reconstructed = dict(mean=float(b.mean()), sd=float(b.std(ddof=1)), q025=float(np.quantile(b,.025)), q975=float(np.quantile(b,.975)), mcse_mean=float(az.mcse(idata,method='mean').beta), rhat=float(az.rhat(idata,method='rank').beta), ess_bulk=float(az.ess(idata,method='bulk').beta), ess_tail=float(az.ess(idata,method='tail').beta))
    for key, value in reconstructed.items():
        np.testing.assert_allclose(value, result['beta'][key], rtol=1e-12, atol=1e-12)
    summaries[stage] = reconstructed
print(json.dumps({'preserved_files':len(snapshot), 'checks':'source-only beta change; IR-only beta scale change; unchanged data/settings/backend; native fingerprint and NPZ equality; same result schema; beta reconstructed', 'beta':summaries}, indent=2))
if '--record' in sys.argv:
    paths = [Path(p) for p in ['model_revised.py','analysis_revised.py','run_revised.sh','verify_revision.py','preservation-before-revision.json','STUDY.md','STUDY-initial.md','provenance.json']]
    paths += [p for stage in ['revised','revised-prior','revised-recovery'] for p in (Path('results')/stage).rglob('*') if p.is_file()]
    paths += [p for p in Path('logs/revised').rglob('*') if p.is_file() and p.name != 'verification.log']
    record = {'status':'scripted revised proposal; awaits human scientific review', 'initial_inventory':'provenance.json', 'preservation_snapshot':'preservation-before-revision.json', 'preservation_verified_files':len(snapshot), 'sha256':{str(p):sha(p) for p in sorted(paths)}, 'versions':{k:md.version(k) for k in ['bayeswire','bayesjax','bayescycle','jax','jaxlib','blackjax','arviz','numpy','matplotlib']}, 'beta_reconstructed':summaries}
    with Path('revision-provenance.json').open('x') as f:
        json.dump(record,f,indent=2)
else:
    if Path('revision-provenance.json').exists():
        for path, digest in json.loads(Path('revision-provenance.json').read_text())['sha256'].items():
            assert sha(path) == digest, path
        print('Revision inventory verified.')
