# File-only handoff

## Question, proposal and approval

Estimate beta, the expected within-clinic outcome difference for a one-unit increase in x. This is a conditional association, not an identified causal effect. `input/data.json` contains finite x/y values for 160 observations and a 160-by-8 clinic design (20 observations per clinic).

The current proposed model is `model_revised.py`:

- alpha ~ Normal(0, 2); **beta ~ Normal(0, 0.25)** (SD 0.25).
- tau and sigma ~ HalfNormal(1); eight independent z_j ~ Normal(0, 1).
- y_i ~ Normal(alpha + tau*(clinic_design @ z)_i + beta*x_i, sigma).

It assumes a common linear slope, exchangeable Gaussian clinic intercepts, homoskedastic Gaussian residuals and conditional independence, treating x/design as fixed. Eight clinics weakly inform the population intercept scale. There is no causal identification, random-slope, measurement-error or missingness model.

`STUDY.md`, the executable model and `results/revision-provenance.json` identify this as a scripted proposal awaiting human review. **Nothing is documented as human scientifically approved.** Completed runs, evaluator requests and passing numerical diagnostics are not approvals.

Current outputs are `results/revised/`, `results/revised-prior/` and `results/revised-recovery/`. `model.py` has the superseded Normal(0, 1) beta prior; `results/initial/`, `results/prior/`, `results/recovery/` and the original top-level provenance/execution/COMPLETE describe that historical stage. The historical tail of `STUDY.md` saying no artifacts are superseded is explicitly scoped to that initial stage, not current status. Comparing executable models confirms only beta's prior scale changed (apart from the docstring).

## Independent posterior reconstruction

Loaded `results/revised/posterior.npz`, named beta array, shape (4 chains, 1000 retained draws). No model was imported or run. NumPy 2.5.2 and ArviZ 0.22.0 recomputation gives:

| Metric | Recomputed value |
|---|---:|
| Mean | 0.5613208509433069 |
| SD (ddof=1) | 0.057128674571364925 |
| 2.5% quantile | 0.44791227303739845 |
| 97.5% quantile | 0.6692952138207736 |
| Rank Rhat | 1.0028415014241625 |
| Bulk ESS | 2905.4334765576864 |
| Tail ESS | 2336.3854408136567 |
| MCSE(mean) | 0.0010613965814576546 |

All eight metrics exactly match `results/revised/result.json` (maximum absolute difference 0); they also agree with the rounded current table in `STUDY.md`. The JSON interval is equal-tailed 95%; the diagnostic CSV's default ArviZ HDI columns are 94%, not interchangeable intervals.

The runner `analysis_revised.py` records NumPyro NUTS, float64, four parallel CPU chains, 500 warmup, 1000 retained draws/chain, target acceptance 0.9, maximum depth 10, supplied-data seed 4201. `results/revision-execution.json` records prior prediction, recovery, supplied-data fitting, two fit attempts and zero retries; this handoff did not replay execution.

## Provenance checks and limits

Recomputed SHA256 from exact saved bytes:

- `results/revision-provenance.json`: all seven in-arm source/input entries match, including both model/runner versions and all three input files.
- `results/provenance.json`: all five in-arm source/input entries match.
- `results/revision-provenance.json`'s `preserved_initial_sha256`: all 23 retained initial files match, including initial posterior artifacts and original provenance.
- No mismatches among checked entries. External `../INITIAL-BRIEF.md`, `../REVISION-BRIEF.md` and `../PROTOCOL.md` were deliberately not read or hashed under the arm-files-only restriction; their recorded hashes remain unverified.

The revised model SHA256 is `5ae8f48b5cc24122be60c5818525bd0c2b03848dad92b6321f910423e56ac961`; supplied input SHA256 is `8a753a950786ff60f47747dd0176ff01619d2f958402e0d042299fcb4523018d`.

These are unsigned local manifests, not independent attestations. No recorded revised-output digest binds the current posterior to the revised model/input execution. Summary agreement demonstrates internal numerical consistency, not that those draws were necessarily produced by that execution. Source hashes prove byte identity only: they do not establish deterministic arbitrary Python execution, identical dependencies/runtime state, or independently verify historical run counts. Recorded versions/settings/seeds aid reproduction but do not prove it. No refit was authorized or performed.

## Diagnostics and next human decision

Independently checked all parameter components and saved sampler statistics in both current fits:

| Fit | Maximum rank Rhat | Minimum bulk ESS | Minimum tail ESS | Divergences | Maximum steps |
|---|---:|---:|---:|---:|---:|
| Revised supplied data | 1.00627068 | 730.75997 | 1072.58895 | 0 | 63 |
| Revised recovery | 1.00588254 | 832.38377 | 1335.36837 | 0 | 79 |

Neither fit has saved step counts >=1023. These support sampling reliability, not scientific validity or a guarantee of convergence. The saved recovery result (`results/revised-recovery/result.json`) places truth beta=0.4 outside [0.15277926, 0.37213862]; other supplied component truths are covered. One fixture is not a calibration assessment, and the miss should not automatically trigger tuning.

`STUDY.md` reports visual inspection of revised trace/prior/posterior-predictive PNGs by the preceding agent; this handoff has not independently inspected the images. Reported aggregate predictive overlap does not test clinic-specific residuals or establish adequacy. Predictions condition on existing clinics, use only 200 draws and are not new-clinic predictions. The prior can still produce extreme outcomes.

Next: a human must decide whether SD 0.25 is scientifically justified and whether to retain this proposal, interpret the recovery miss, review assumptions/estimand and determine whether additional residual or recovery checks are warranted. Stop pending that decision; no study advancement occurred.

## Reproducible read-only calculation

From this directory, use the following command. `UV_NO_SYNC=1` avoids dependency synchronization; it does not run either analysis script or regenerate simulations.

```sh
UV_NO_SYNC=1 uv run --project .. python - <<'PY'
import hashlib, json
from pathlib import Path
import numpy as np
import arviz as az
with np.load('results/revised/posterior.npz', allow_pickle=False) as f:
    posterior = {k: f[k] for k in f.files}
b = posterior['beta']
d = az.from_dict(posterior=posterior)
m = dict(mean=float(b.mean()), sd=float(b.std(ddof=1)),
         q025=float(np.quantile(b, .025)), q975=float(np.quantile(b, .975)))
for key, fn, method in [('rhat', az.rhat, 'rank'),
                        ('ess_bulk', az.ess, 'bulk'),
                        ('ess_tail', az.ess, 'tail'),
                        ('mcse_mean', az.mcse, 'mean')]:
    m[key] = float(fn(d, method=method).beta)
s = json.loads(Path('results/revised/result.json').read_text())['beta']
print(np.__version__, az.__version__, b.shape)
print(json.dumps(m, indent=2))
print('summary_matches', all(np.isclose(m[k], s[k], rtol=1e-12, atol=1e-12) for k in m))
print('max_abs_difference', max(abs(m[k]-s[k]) for k in m))
for name in ['results/provenance.json', 'results/revision-provenance.json']:
    manifest = json.loads(Path(name).read_text())
    for section in ['sha256', 'preserved_initial_sha256']:
        for file, expected in manifest.get(section, {}).items():
            if '..' in Path(file).parts:
                print(name, section, file, 'OUTSIDE ARM: NOT CHECKED')
                continue
            actual = hashlib.sha256(Path(file).read_bytes()).hexdigest()
            print(name, section, file, actual == expected)
for stage in ['revised', 'revised-recovery']:
    with np.load(f'results/{stage}/posterior.npz') as f:
        d = az.from_dict(posterior={k: f[k] for k in f.files})
    print(stage, 'max_rhat', max(float(v.max()) for v in az.rhat(d, method='rank').data_vars.values()),
          'min_bulk', min(float(v.min()) for v in az.ess(d, method='bulk').data_vars.values()),
          'min_tail', min(float(v.min()) for v in az.ess(d, method='tail').data_vars.values()))
    with np.load(f'results/{stage}/sampler_stats.npz') as f:
        print('divergences', int(f['diverging'].sum()),
              'max_steps', int(f['num_steps'].max()),
              'tree_limit', int((f['num_steps'] >= 1023).sum()))
PY
```
