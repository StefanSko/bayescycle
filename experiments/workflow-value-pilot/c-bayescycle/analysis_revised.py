"""Supplementary artifact conversion, ArviZ 0.22 checks, recovery and PPC.
No inference here. CLI performs all prior simulation and fitting.
Commands: uv run --project .. python analysis_revised.py prior|recovery|revised
Revision-only copy of analysis.py; same diagnostic and predictive definitions.
Refuses existing derived outputs. Native CLI artifacts remain unchanged.
"""
import hashlib
import json
import sys
from pathlib import Path
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

assert az.__version__.startswith('0.22.')

def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_npz(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)

def predictive_plot(yrep, y, path, title):
    if path.exists():
        raise FileExistsError(path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    grid = np.linspace(min(yrep.min(), y.min()), max(yrep.max(), y.max()), 45)
    for row in yrep[:40]:
        axes[0].hist(row, bins=grid, density=True, histtype='step', color='C0', alpha=.12)
    axes[0].hist(y, bins=grid, density=True, histtype='step', color='black', lw=2)
    axes[0].set(xlabel='y', title='Replicates (blue), observed (black)')
    for ax, fun, label in [(axes[1], np.mean, 'Mean'), (axes[2], np.std, 'SD')]:
        ax.hist(fun(yrep, axis=1), bins=25, color='C0', alpha=.7)
        ax.axvline(fun(y), color='black', lw=2)
        ax.set(xlabel=label, ylabel='Replicate count')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def prior():
    run = Path('results/revised-prior')
    rows = [json.loads(line) for line in (run/'prior_predictive.ndjson').read_text().splitlines()]
    assert rows[0]['prior_predictive_format'] == 'v0-provisional'
    draws = [r['values'] for r in rows[1:] if 'values' in r]
    assert len(draws) == 200
    arrays = {k: np.asarray([r[k] for r in draws], dtype=np.float64) for k in draws[0]}
    save_npz(run/'prior_predictive.npz', **arrays)
    y = np.asarray(json.loads(Path('input/data.json').read_text())['y'])
    predictive_plot(arrays['y'], y, run/'prior_predictive.png', 'Prior predictive: 200 draws, seed 4200')
    dump(run/'predictive_summary.json', {'y_quantiles': np.quantile(arrays['y'], [.025,.5,.975]).tolist(), 'finite': bool(np.isfinite(arrays['y']).all())})

def fit(stage):
    run = Path('results/revised-recovery' if stage == 'recovery' else 'results/revised')
    if (run/'posterior.npz').exists() or (run/'result.json').exists():
        raise FileExistsError('Completed derived result exists; no overwrite')
    rows = [json.loads(line) for line in (run/'posterior.ndjson').read_text().splitlines()]
    header, trailer = rows[0], rows[-1]['trailer']
    assert header['draws_format'] == trailer['draws_format'] == 'v0-provisional'
    fingerprint = 'sha256:' + hashlib.sha256(b'bayescycle-model-data-v1\n' + (run/'model.ir.json').read_bytes() + b'\n' + (run/'data.json').read_bytes()).hexdigest()
    assert header['model_data_fingerprint'] == trailer['model_data_fingerprint'] == fingerprint
    draws = rows[1:-1]
    assert len(draws) == 4000
    samples = {p['name']: np.empty((4,1000,*p['shape']), dtype=np.float64) for p in header['params']}
    seen = set()
    stats = {k: np.empty((4,1000)) for k in ('diverging','tree_depth','tree_accept','energy')}
    for r in draws:
        assert r['draws_format'] == 'v0-provisional'
        idx = (r['chain'],r['draw'])
        assert idx not in seen
        seen.add(idx)
        for k in samples: samples[k][idx] = r['values'][k]
        for k in stats: stats[k][idx] = r[k]
    assert len(seen) == 4000 and all(np.isfinite(a).all() for a in samples.values())
    save_npz(run/'posterior.npz', **samples)
    save_npz(run/'sample_stats.npz', **stats)
    idata = az.from_dict(posterior=samples)
    rh, eb, et, mc = az.rhat(idata, method='rank'), az.ess(idata, method='bulk'), az.ess(idata, method='tail'), az.mcse(idata, method='mean')
    summaries = {}
    for k, a in samples.items():
        summaries[k] = {key: np.asarray(value).tolist() for key,value in {
            'mean': a.mean(axis=(0,1)), 'sd': a.std(axis=(0,1), ddof=1),
            'q025': np.quantile(a,.025,axis=(0,1)), 'q975': np.quantile(a,.975,axis=(0,1)),
            'mcse_mean': mc[k].values, 'rhat': rh[k].values, 'ess_bulk': eb[k].values, 'ess_tail': et[k].values}.items()}
    flags = [k for k in samples if np.any(rh[k].values>1.01) or np.any(eb[k].values<400) or np.any(et[k].values<400)]
    original = Path('input/recovery.json' if stage=='recovery' else 'input/data.json')
    result = {'beta': summaries['beta'], 'parameters': summaries,
        'divergences': int(stats['diverging'].sum()), 'diagnostic_flagged_parameters': flags,
        'sampler': {'backend':'bayesjax', 'algorithm':'BlackJAX NUTS', 'seed':header['seed'], 'chains':4, 'dtype':'float64', **header['settings']},
        'arviz_version':az.__version__, 'samples':str(run/'posterior.npz'),
        'native_samples':str(run/'posterior.ndjson'), 'model_data_fingerprint': fingerprint,
        'sha256': {str(p):sha(p) for p in [Path('model_revised.py'), original, run/'model.ir.json', run/'data.json']}}
    if stage=='recovery':
        truth = json.loads(Path('input/recovery-truth.json').read_text())
        result['recovery'] = {k:{'truth':v, 'covered_95': ((np.asarray(v)>=summaries[k]['q025']) & (np.asarray(v)<=summaries[k]['q975'])).tolist()} for k,v in truth.items()}
        result['sha256']['input/recovery-truth.json'] = sha('input/recovery-truth.json')
    else:
        data = json.loads(original.read_text())
        rng = np.random.default_rng(4202)
        indices = rng.choice(4000, size=200, replace=False)
        p = {k:a.reshape((-1,*a.shape[2:]))[indices] for k,a in samples.items()}
        mu = p['alpha'][:,None] + p['tau'][:,None]*(p['z'] @ np.asarray(data['clinic_design']).T) + p['beta'][:,None]*np.asarray(data['x'])
        yrep = rng.normal(mu, p['sigma'][:,None])
        save_npz(run/'posterior_predictive.npz', y=yrep, mu=mu, posterior_flat_indices=indices)
        predictive_plot(yrep, np.asarray(data['y']), run/'posterior_predictive.png', 'Posterior predictive: 200 draws, seed 4202')
        result['posterior_predictive'] = {'seed':4202,'draws':200,'method':'NumPy conditional Normal simulation; same observed clinic design','samples':str(run/'posterior_predictive.npz')}
    az.plot_trace(idata, compact=True)
    plt.gcf().savefig(run/'trace.png', dpi=120, bbox_inches='tight')
    plt.close('all')
    result['figures'] = [str(run/'trace.png')] + ([str(run/'posterior_predictive.png')] if stage=='revised' else [])
    dump(run/'result.json', result)
    print(json.dumps({'stage':stage,'beta':result['beta'],'divergences':result['divergences'],'flags':flags}, indent=2))

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'prior': prior()
    elif stage in ('revised','recovery'): fit(stage)
    else: raise ValueError(stage)
