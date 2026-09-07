"""Run once: uv run --project .. python analysis.py (from this directory)."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import jax
import numpy as np
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bayeswire.ir import canonical_bytes
from bayeswire.model import model_meta
from bayesjax import bind_model
from bayesjax.inference import sample
from bayesjax.simulation import simulate_prior_predictive
from bayesjax.diagnostics import rhat, ess
from model import ClinicModel

ROOT = Path(__file__).resolve().parent
SETTINGS = dict(num_chains=4, num_warmup=500, num_samples=1000,
                target_acceptance_rate=0.9, max_tree_depth=10)

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def load(name):
    return {k: np.asarray(v, dtype=np.float64) for k, v in json.loads((ROOT / name).read_text()).items()}

def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def predictive_plot(yrep, data, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for row in yrep[:40]:
        axes[0].hist(row, bins=25, density=True, histtype='step', alpha=.12, color='C0')
    axes[0].hist(data['y'], bins=25, density=True, histtype='step', color='black', lw=2, label='observed')
    axes[0].legend(); axes[0].set_xlabel('y'); axes[0].set_ylabel('density')
    for ax, fun, label in [(axes[1], np.mean, 'mean(y)'), (axes[2], np.std, 'SD(y)')]:
        ax.hist(fun(yrep, axis=1), bins=25)
        ax.axvline(fun(data['y']), color='black', label='observed')
        ax.set_xlabel(label)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def fit(label, input_name, seed, provenance):
    folder = ROOT / 'results' / label
    folder.mkdir(exist_ok=False)
    data = load(input_name)
    started = time.perf_counter()
    result = sample(bind_model(ClinicModel, data), seed=seed, **SETTINGS)
    draws = {k: np.asarray(v) for k, v in result.samples.items()}
    seconds = time.perf_counter() - started
    np.savez_compressed(folder / 'posterior.npz', **draws)
    native = {}
    for phase in ['warmup', 'sampling']:
        for name in ['is_divergent', 'acceptance_rate', 'num_integration_steps', 'num_trajectory_expansions', 'energy']:
            native[phase + '_' + name] = np.asarray(getattr(getattr(result.diagnostics, phase), name))
    native['step_size'] = np.asarray(result.adaptation.step_size)
    np.savez_compressed(folder / 'native_diagnostics.npz', **native)
    idata = az.from_dict(posterior=draws, sample_stats={'diverging': native['sampling_is_divergent']})
    table = az.summary(idata, kind='all', round_to='none')
    table.to_csv(folder / 'arviz_summary.csv')
    b = draws['beta'].reshape(-1)
    beta = dict(mean=float(b.mean()), sd=float(b.std(ddof=1)),
                q025=float(np.quantile(b,.025)), q975=float(np.quantile(b,.975)))
    beta.update({k: float(table.loc['beta', k]) for k in ['mcse_mean','r_hat','ess_bulk','ess_tail']})
    beta['rhat'] = beta.pop('r_hat')
    az.plot_trace(idata, var_names=['alpha','beta','tau','sigma','z'], compact=True)
    plt.gcf().set_size_inches(13,12); plt.tight_layout(); plt.savefig(folder / 'trace.png', dpi=130); plt.close('all')
    # Ordinary Python posterior prediction, conditional on the same fixed design.
    predictive_seed = 4202 if label == 'initial' else 4302
    rng = np.random.default_rng(predictive_seed)
    idx = rng.choice(4000, size=200, replace=False)
    flat = {k: v.reshape((-1,) + v.shape[2:])[idx] for k,v in draws.items()}
    mu = flat['alpha'][:,None] + flat['tau'][:,None] * (flat['z'] @ data['clinic_design'].T) + flat['beta'][:,None]*data['x']
    yrep = rng.normal(mu, flat['sigma'][:,None])
    np.savez_compressed(folder / 'posterior_predictive.npz', y=yrep, mu=mu, posterior_flat_indices=idx)
    predictive_plot(yrep, data, folder / 'posterior_predictive.png', label + ': posterior predictive (200 draws)')
    flags = {name: list(table.index[mask]) for name, mask in {
        'rank_rhat_gt_1.01': table.r_hat > 1.01,
        'bulk_ess_lt_400': table.ess_bulk < 400,
        'tail_ess_lt_400': table.ess_tail < 400}.items()}
    out = dict(beta=beta, divergences=int(native['sampling_is_divergent'].sum()),
               warmup_divergences=int(native['warmup_is_divergent'].sum()),
               sampler=dict(backend='bayesjax / BlackJAX NUTS', seed=seed, dtype='float64', **SETTINGS),
               sampling_wall_seconds=seconds, diagnostic_flags=flags,
               native_classic_split_rhat={k:float(v) for k,v in rhat(result.samples).items()},
               native_split_geyer_ess={k:float(v) for k,v in ess(result.samples).items()},
               posterior_predictive=dict(seed=predictive_seed, draws=200, method='NumPy conditional Normal draws; uniform posterior subsample without replacement'),
               paths=dict(samples=f'results/{label}/posterior.npz', figures=[f'results/{label}/trace.png',f'results/{label}/posterior_predictive.png'], native_diagnostics=f'results/{label}/native_diagnostics.npz', prior_predictive='results/prior/prior_predictive.png'),
               provenance=provenance)
    if label == 'recovery':
        truth = json.loads((ROOT / 'input/recovery-truth.json').read_text())
        coverage = {}
        for k,v in draws.items():
            q = np.quantile(v, [.025,.975], axis=(0,1))
            coverage[k] = dict(truth=truth[k], q025=q[0].tolist(), q975=q[1].tolist(), covered=((q[0] <= truth[k]) & (truth[k] <= q[1])).tolist())
        out['recovery'] = coverage
    save_json(folder / 'result.json', out)
    (folder / 'COMPLETE').write_text('Completed. Recovery was repeated with the same seed after a diagnostic-export failure; see STUDY.md.\n')
    print(label, json.dumps(out), flush=True)

def main():
    started = time.perf_counter()
    assert jax.config.x64_enabled, 'float64 required'
    assert az.__version__.startswith('0.22'), az.__version__
    # Stronger than refusing completed outputs: refuse any existing stage directory.
    for stage in ['prior','recovery','initial']:
        if (ROOT / 'results' / stage).exists():
            raise FileExistsError(f'refusing overwrite: results/{stage}')
    (ROOT / 'results').mkdir(exist_ok=True)
    wire = canonical_bytes(model_meta(ClinicModel))
    (ROOT / 'results/model-ir.json').write_bytes(wire)
    files = ['model.py','analysis.py','input/data.json','input/recovery.json','input/recovery-truth.json','results/model-ir.json']
    provenance = dict(sha256={p:hash_file(ROOT/p) for p in files},
        versions={p:importlib.metadata.version(p) for p in ['bayeswire','bayesjax','jax','jaxlib','blackjax','arviz','numpy']},
        jax_devices=[str(d) for d in jax.devices()])
    save_json(ROOT / 'results/provenance.json', provenance)
    prior_dir = ROOT / 'results/prior'; prior_dir.mkdir()
    data = load('input/data.json')
    prior = simulate_prior_predictive(ClinicModel, seed=4200, num_samples=200,
        data={k:v for k,v in data.items() if k != 'y'}, observed_shapes={'y':data['y'].shape})
    np.savez_compressed(prior_dir / 'prior_predictive.npz', y=np.asarray(prior.observed['y']), **{k:np.asarray(v) for k,v in prior.parameters.items()})
    predictive_plot(np.asarray(prior.observed['y']), data, prior_dir / 'prior_predictive.png', 'Prior predictive: 200 draws, seed 4200')
    save_json(prior_dir / 'result.json', dict(seed=4200, num_samples=200, provenance=provenance))
    (prior_dir / 'COMPLETE').write_text('Prior prediction completed before either fit.\n')
    print('prior prediction complete', flush=True)
    fit('recovery','input/recovery.json',4301,provenance)
    fit('initial','input/data.json',4201,provenance)
    save_json(ROOT / 'results/execution.json', dict(total_script_wall_seconds=time.perf_counter()-started, order=['prior','recovery','initial'], fit_attempts_this_invocation=2, prior_failed_invocations=1, recovery_same_seed_repair_retries=1))

if __name__ == '__main__':
    main()
