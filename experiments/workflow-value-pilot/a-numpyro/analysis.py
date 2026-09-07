"""Run once: uv run --project .. python analysis.py
Refuses an existing results directory (including partial attempts).
"""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')
import hashlib
import json
import platform
import time
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from model import model

ROOT = Path(__file__).resolve().parent
PARAMS = ['alpha', 'beta', 'tau', 'z', 'sigma']
SETTINGS = dict(chains=4, warmup=500, draws_per_chain=1000,
                target_acceptance=0.9, max_tree_depth=10,
                chain_method='parallel', backend='NumPyro NUTS', float_dtype='float64')


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load(name):
    return {k: jnp.asarray(v, dtype=jnp.float64)
            for k, v in json.loads((ROOT / 'input' / name).read_bytes()).items()}


def predictive_figure(yrep, observed, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    bins = np.linspace(min(yrep.min(), observed.min()), max(yrep.max(), observed.max()), 45)
    for row in yrep[:40]:
        axes[0].hist(row, bins=bins, density=True, histtype='step', color='C0', alpha=.12)
    axes[0].hist(observed, bins=bins, density=True, histtype='step', color='black', linewidth=2)
    axes[0].set(title='Outcome distributions (40 replicates)', xlabel='y')
    for ax, stat, label in [(axes[1], np.mean, 'Mean'), (axes[2], np.std, 'SD (ddof=0)')]:
        ax.hist(stat(yrep, axis=1), bins=25, color='C0', alpha=.7)
        ax.axvline(stat(observed), color='black', label='Observed')
        ax.set(title=f'Replicate {label}', xlabel=label)
        ax.legend()
    fig.suptitle(title + ' — fixed observed x/design')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fit(data, stage, seed, truth=None):
    directory = ROOT / 'results' / stage
    directory.mkdir()
    start = time.perf_counter()
    sampler = MCMC(NUTS(model, target_accept_prob=.9, max_tree_depth=10),
                   num_warmup=500, num_samples=1000, num_chains=4,
                   chain_method='parallel', progress_bar=False)
    sampler.run(jax.random.PRNGKey(seed), **data,
                extra_fields=('diverging', 'num_steps', 'accept_prob', 'potential_energy'))
    posterior = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
    extras = {k: np.asarray(v) for k, v in sampler.get_extra_fields(group_by_chain=True).items()}
    elapsed = time.perf_counter() - start
    np.savez_compressed(directory / 'posterior.npz', **posterior)
    np.savez_compressed(directory / 'sampler_stats.npz', **extras)
    idata = az.from_dict(posterior=posterior, sample_stats={'diverging': extras['diverging']})
    summary = az.summary(idata, var_names=PARAMS, kind='all', round_to='none')
    summary.to_csv(directory / 'diagnostics.csv')
    beta = posterior['beta']
    beta_summary = dict(mean=float(beta.mean()), sd=float(beta.std(ddof=1)),
                        q025=float(np.quantile(beta, .025)), q975=float(np.quantile(beta, .975)))
    for key, val in [('mcse_mean', az.mcse(idata, method='mean')),
                     ('rhat', az.rhat(idata, method='rank')),
                     ('ess_bulk', az.ess(idata, method='bulk')),
                     ('ess_tail', az.ess(idata, method='tail'))]:
        beta_summary[key] = float(val.beta)
    failures = []
    for name, row in summary.iterrows():
        if row.r_hat > 1.01 or row.ess_bulk < 400 or row.ess_tail < 400:
            failures.append(dict(parameter=name, rhat=float(row.r_hat),
                                 ess_bulk=float(row.ess_bulk), ess_tail=float(row.ess_tail)))
    divergences = int(extras['diverging'].sum())
    plt.figure()
    az.plot_trace(idata, var_names=['alpha', 'beta', 'tau', 'sigma', 'z'], compact=True)
    plt.gcf().tight_layout()
    plt.gcf().savefig(directory / 'trace.png', dpi=130)
    plt.close('all')
    result = dict(beta=beta_summary, divergences=divergences,
                  sampler={**SETTINGS, 'seed': seed, 'fit_wall_seconds': elapsed,
                           'mean_accept_prob': float(extras['accept_prob'].mean()),
                           'max_num_steps': int(extras['num_steps'].max()),
                           'steps_at_tree_limit': int((extras['num_steps'] >= 1023).sum())},
                  diagnostic_flags=failures, diagnostic_pass=not failures and divergences == 0,
                  diagnostics_definition='ArviZ 0.22 rank Rhat, bulk/tail ESS; flags: Rhat >1.01, ESS <400, any divergences',
                  paths={'samples': f'results/{stage}/posterior.npz',
                         'sampler_stats': f'results/{stage}/sampler_stats.npz',
                         'diagnostics': f'results/{stage}/diagnostics.csv',
                         'trace': f'results/{stage}/trace.png'},
                  provenance='results/provenance.json')
    if truth is not None:
        coverage = {}
        for name in PARAMS:
            low, high = np.quantile(posterior[name], [.025, .975], axis=(0, 1))
            target = np.asarray(truth[name])
            coverage[name] = dict(truth=target.tolist(), mean=posterior[name].mean(axis=(0, 1)).tolist(),
                                  q025=low.tolist(), q975=high.tolist(),
                                  covered=((target >= low) & (target <= high)).tolist())
        result['recovery'] = coverage
        result['recovery_interpretation'] = 'One fixture; descriptive coverage, not a calibration pass/fail test.'
    if stage == 'initial':
        # Random subset without replacement; independent reproducible split from seed 4202.
        select_key, predictive_key = jax.random.split(jax.random.PRNGKey(4202))
        indices = np.asarray(jax.random.choice(select_key, 4000, shape=(200,), replace=False))
        selected = {k: jnp.asarray(v.reshape((-1,) + v.shape[2:])[indices]) for k, v in posterior.items()}
        draws = np.asarray(Predictive(model, posterior_samples=selected, return_sites=['y'])(
            predictive_key, x=data['x'], clinic_design=data['clinic_design'])['y'])
        np.savez_compressed(directory / 'posterior_predictive.npz', y=draws, flat_posterior_indices=indices)
        predictive_figure(draws, np.asarray(data['y']), directory / 'posterior_predictive.png', 'Posterior predictive')
        result['posterior_predictive'] = dict(seed=4202, draws=200, index_order='chain-major flatten')
        result['paths'].update(posterior_predictive='results/initial/posterior_predictive.npz',
                               posterior_predictive_figure='results/initial/posterior_predictive.png',
                               prior_predictive_figure='results/prior/prior_predictive.png')
    save_json(directory / 'result.json', result)
    (directory / 'COMPLETE').write_text('Completed; do not overwrite.\n')
    print(stage, json.dumps(result), flush=True)


def main():
    start = time.perf_counter()
    assert jax.config.x64_enabled, 'float64 required'
    assert jax.local_device_count() >= 4, 'four devices required'
    assert az.__version__.startswith('0.22.'), 'ArviZ 0.22 required'
    results = ROOT / 'results'
    results.mkdir(exist_ok=False)
    files = ['model.py', 'analysis.py', 'input/data.json', 'input/recovery.json', 'input/recovery-truth.json',
             '../INITIAL-BRIEF.md', '../PROTOCOL.md']
    save_json(results / 'provenance.json', dict(
        sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        versions={'python': platform.python_version(), 'jax': jax.__version__,
                  'numpyro': numpyro.__version__, 'arviz': az.__version__, 'numpy': np.__version__},
        devices=[str(d) for d in jax.devices()],
        environment={k: os.environ.get(k) for k in ['JAX_ENABLE_X64', 'MPLBACKEND', 'OMP_NUM_THREADS', 'XLA_FLAGS']},
        command='uv run --project .. python analysis.py', model_stage='initial',
        status='Engineering fixture; awaits human scientific review'))
    data = load('data.json')
    prior_dir = results / 'prior'
    prior_dir.mkdir()
    prior = {k: np.asarray(v) for k, v in Predictive(model, num_samples=200)(
        jax.random.PRNGKey(4200), x=data['x'], clinic_design=data['clinic_design']).items()}
    np.savez_compressed(prior_dir / 'prior_predictive.npz', **prior)
    predictive_figure(prior['y'], np.asarray(data['y']), prior_dir / 'prior_predictive.png', 'Prior predictive')
    save_json(prior_dir / 'result.json', dict(seed=4200, draws=200, provenance='results/provenance.json'))
    (prior_dir / 'COMPLETE').write_text('Completed before recovery and supplied-data fitting.\n')
    print('Prior prediction complete', flush=True)
    fit(load('recovery.json'), 'recovery', 4301, json.loads((ROOT / 'input/recovery-truth.json').read_bytes()))
    fit(data, 'initial', 4201)
    save_json(results / 'execution.json', dict(total_wall_seconds=time.perf_counter() - start,
                                             stage_order=['prior', 'recovery', 'initial'], fit_attempts=2,
                                             retries=0))
    (results / 'COMPLETE').write_text('Initial stage complete; no revision or handoff executed.\n')


if __name__ == '__main__':
    main()
