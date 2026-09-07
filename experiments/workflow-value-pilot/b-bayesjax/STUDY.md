# Arm B — scripted prior revision complete

**Current proposed work: `model_revised.py`, beta ~ Normal(0,0.25), and
`results/revised/`. All work awaits human scientific review.**
This revision is an evaluator-fixed request, not scientific approval. No approval
gates were invoked or satisfied. No Bayescycle CLI, internals, alternate backend,
dependency installation, shared-input changes or source-library changes were used.
The separate fresh-session handoff stage remains pending.

## Current revision and supersession

Only beta's prior changed scientifically: its SD is now 0.25 rather than 1.
All likelihood terms, other priors, constraints, design and input bytes remain
unchanged. `model.py` remains the exact initial executable declaration;
`analysis.py` and every initial result remain untouched. The revised declaration
is `model_revised.py`; its only other difference is its status docstring.

The initial prior prediction, recovery coverage, posterior fits, diagnostics and
predictive figures in `results/prior/`, `results/recovery/` and `results/initial/`
are **superseded for the current proposal**, not erased or invalidated as historical
computations. The original provenance and canonical IR describe only that initial
model. The question, assumptions and limitations below still apply.

Current outputs:
- `results/revised-prior/`: 200 public Bayesjax prior predictive draws, seed 4200.
- `results/revised-recovery/`: separate recovery fit, seed 4301, coverage and plots.
- `results/revised/posterior.npz` and `results/revised/result.json`: supplied-data
  fit, seed 4201, using the same schema as initial; full ArviZ CSV, native warmup
  and sampling diagnostics, trace and posterior predictive artifacts alongside.
- `results/revised-model-ir.json`, `results/revised-provenance.json`: canonical
  revised model and hashes of executable source, unchanged input bytes and IR,
  installed versions and four CPU devices. Provenance is embedded in summaries.

Both fits again used float64, Bayesjax / BlackJAX NUTS, four chains, 500 warmup,
1000 retained draws per chain, target acceptance 0.9, max tree depth 10. No
retuning or seed changes occurred. Supplied/recovery posterior prediction uses
200 draws with seeds 4202/4302 respectively, using the unchanged ordinary NumPy
conditional predictive method, not a backend predictive API.

| Revised dataset | beta mean | SD | equal-tailed 95% interval | MCSE mean | rank Rhat | bulk / tail ESS |
|---|---:|---:|---|---:|---:|---:|
| supplied | 0.561162 | 0.056389 | [0.449833, 0.673225] | 0.001115 | 1.001968 | 2562 / 2602 |
| recovery | 0.261660 | 0.053745 | [0.155275, 0.365536] | 0.001026 | 1.000400 | 2752 / 2694 |

Supplied beta mean decreased by about 0.028081 from the preserved initial result.
Neither revised fit has retained-draw divergences or scalar-coordinate flags for
rank Rhat >1.01, bulk ESS <400 or tail ESS <400. Warmup divergences were **27
supplied and 24 recovery**, recorded separately. Native classic split diagnostics
are retained but are not interchangeable with the independent ArviZ definitions.
Recovery beta truth 0.4 again lies outside the interval; all other generating
parameters and z coordinates are covered. This is one descriptive fixture, not
a calibration test or evidence that either prior is scientifically preferable.

### Revision plots actually inspected

This revision agent opened `results/revised-prior/prior_predictive.png`,
`results/revised/trace.png` and `results/revised/posterior_predictive.png`:
- Prior predictive means and SDs span the observed values; prior outcomes remain
  broad despite the narrower beta prior because other uncertainty is unchanged.
- Chain densities broadly overlap without obvious persistent separation; tau has
  a right tail and excursions. Compact z traces remain crowded.
- Observed marginal histogram overlaps posterior predictions; observed mean and
  SD lie near the center of their predictive distributions.
Revised recovery plots are saved but were not visually inspected. These marginal
checks do not assess clinic-specific residual structure or validate assumptions.

### Revision execution, preservation and handoff commands

Executed successfully from this directory:

```sh
uv run --project .. python analysis_revised.py
uv run --project .. python check_revised_artifacts.py
```

The analysis refuses any existing revised stage directory; do not rerun it over
these results. For a fresh replay, copy `model_revised.py`, `analysis_revised.py`,
`revision-preservation.json` and unchanged `input/` into a new empty subdirectory
within this arm, then execute that analysis path with the same command prefix.
Use the supplied environment (float64, four host CPU devices, Agg, OMP threads 1).
For handoff verification without refitting, run only `check_revised_artifacts.py`.
It reconstructs beta statistics and ArviZ diagnostics from NPZ, verifies shapes,
float64, finiteness, divergences, sampler settings, artifact paths and provenance.
`revision-artifact-check.log` records success. The pre-run SHA-256 snapshot
`revision-preservation.json` covers 40 baseline files, including original source,
inputs, every initial result and archived failed-export evidence; all 40 remained
byte-identical after revision. Hashes establish local consistency, not independent
custody. Original `check_artifacts.py` remains available for baseline verification.

`revision-execution.log` and `results/revised-execution.json` retain this run's
record. There were **no revision command failures or retries**: one prior stage
and two fit calls. Successful script body took 6.39 seconds excluding imports;
recovery sampling/materialization 1.74 seconds and supplied fit 1.23 seconds.
End-to-end agent wall time and authoritative tokens/tool counts are harness-owned,
not measured by those script timings. Revision scripts were explicit copies of
initial scripts with model import, stage paths, provenance and status adjusted;
no baseline repair was performed. No human intervention occurred.

### Pending human decisions

Review the scientific justification for the tighter beta prior, the sensitivity
of the estimand, recovery noncoverage, Gaussian/common-slope/exchangeability
assumptions and predictive adequacy before deciding whether to accept, reject or
further revise the proposal. Good MCMC diagnostics are not scientific approval.
No model or result is human-approved.

---

# Preserved initial-stage narrative (historical, superseded)

**Historical status: computational initial stage complete; awaiting human review.**
The sections below describe initial computations, not the current proposed model.
Their numerical summaries are unchanged.

## Question and fixed model

Estimate the within-clinic expected outcome difference for x → x+1: beta.
This is an associational conditional contrast, not an identified causal effect.
Eight clinics, 20 observations each; standardized arbitrary units; fixed x,
complete observations. `model.py` is the Bayeswire declaration:

- alpha ~ Normal(0,2), beta ~ Normal(0,1)
- tau and sigma ~ HalfNormal(1), with positive constraints
- eight independent z_j ~ Normal(0,1)
- y_i ~ Normal(alpha + tau*(clinic_design @ z)_i + beta*x_i, sigma)

Assumptions include exchangeable clinic intercepts, a common linear slope,
conditionally independent Gaussian residuals and constant residual scale.
Only eight groups constrain the group scale; priors matter. Missingness,
measurement error, nonlinearity, heterogeneous slopes, selection and causal
identification are not addressed.

## Execution and results

Order was prior prediction (200 draws, seed 4200), separate recovery fit
(seed 4301), then supplied-data fit (seed 4201). Both fits used public Bayesjax
NUTS with float64, four CPU chains, 500 warmup and 1000 retained draws per chain,
target acceptance 0.9 and maximum tree depth 10. All requested sampler settings
are exposed and were set. No tuning or seed changes were made.

| Dataset | beta mean | SD | equal-tailed 95% interval | MCSE mean | rank Rhat | bulk / tail ESS |
|---|---:|---:|---|---:|---:|---:|
| supplied | 0.589243 | 0.058341 | [0.475300, 0.704704] | 0.001050 | 0.999963 | 3100 / 2455 |
| recovery | 0.273182 | 0.053834 | [0.168704, 0.380117] | 0.000960 | 1.001029 | 3168 / 2437 |

Neither fit has retained-draw divergences or any scalar parameter flagged by
rank Rhat >1.01, bulk ESS <400 or tail ESS <400. There were 25 warmup divergences
for supplied data and 23 for recovery; these are retained separately, not hidden.
ArviZ 0.22 diagnostics are independent of the preserved Bayesjax classic split
Rhat and split Geyer ESS; these definitions are not interchangeable.

**Recovery limitation:** beta's generating truth 0.4 lies outside its 95%
interval. Alpha, sigma, tau and each z coordinate are covered. This single
fixture is descriptive, not a calibration pass/fail test. Good MCMC diagnostics
do not establish model correctness or scientific validity. The noncoverage
was not used to change settings or suppress results.

Posterior prediction uses 200 uniformly selected posterior draws without
replacement and NumPy conditional Normal simulation at the same design,
seed 4202 for supplied data. An extra recovery predictive plot uses seed 4302.
This ordinary Python predictive calculation is explicitly not a Bayesjax
posterior-predictive API. Prior prediction uses Bayesjax's public simulation API.

## Figures actually inspected by this agent

Opened with the image read tool:

- `results/prior/prior_predictive.png`: broad prior predictive location and
  spread; observed mean and SD sit within the simulated ranges. Some prior
  draws produce much wider outcomes than observed. This is not human approval.
- `results/initial/trace.png`: overlapping chain densities and no obvious
  persistent chain separation; tau has a right tail and intermittent excursions.
  The overlaid z traces are crowded and cannot replace coordinate diagnostics.
- `results/initial/posterior_predictive.png`: observed mean and SD are near the
  middle of predictive distributions; marginal histogram broadly overlaps.

Recovery figures were saved but not visually inspected. The plots are limited
marginal checks, not comprehensive clinic-specific residual/model validation.

## Files, provenance and commands

- `results/prior/`: prior parameters, predictive y and figure.
- `results/recovery/` and `results/initial/`: named `(chain,draw,*shape)`
  posterior NPZ, compact result JSON, full ArviZ CSV, trace figure, predictive
  draws/means/selected indices, predictive figure, and native diagnostic NPZ
  containing warmup and retained traces plus adapted step sizes.
- `results/model-ir.json`: exact canonical Bayeswire model bytes used.
- `results/provenance.json`: SHA-256 of **original input bytes**, model source,
  analysis source and canonical IR; installed versions and CPU device list.
  These hashes are also embedded in stage summaries. Inputs were not modified.
- `execution.log`, `artifact-check.log`, `results/execution.json`: execution,
  read-only verification and measured successful-script timing.

From this directory in the supplied environment:

```sh
uv run --project .. python analysis.py
uv run --project .. python check_artifacts.py
```

`analysis.py` intentionally refuses to run if any prior/recovery/initial result
directory exists. For a fresh replay, copy `model.py`, `analysis.py` and unchanged
`input/` into a new empty subdirectory here and execute its analysis script
from this arm directory with the same `uv run --project .. python` prefix.
Do not delete or overwrite these completed results. Supplied environment requires
JAX_ENABLE_X64=true, MPLBACKEND=Agg, four host CPU devices, OMP_NUM_THREADS=1.
The checker does not refit: it verifies hashes, shapes, dtype, finiteness,
recomputes beta diagnostics from saved draws and compares saved divergences.
It was executed successfully. No externally trusted signature authenticates the
manifest; hashes establish local consistency, not independent custody.

## Friction and timing

One initial invocation completed prior prediction and recovery sampling but
failed while exporting diagnostics: the script wrongly called
`dataclasses.fields` on a diagnostic trace. The whole attempt was moved intact
to `scratch-export-failure/`, preserving the completed prior outputs, recovery
posterior and error log. The original script was reconstructed there and its
SHA-256 verified against that attempt's manifest. This scratch script is for
inspection, not execution. Recovery native diagnostics were not saved in that
failed attempt.

The corrected run explicitly repeated recovery with **the same seed and settings**;
`check_artifacts.py` confirms every recovery posterior value is exactly identical
to the failed-export attempt. Thus three sampler executions occurred in total
(two recovery, one supplied-data); no favorable-seed search occurred. Prior
prediction was also repeated with the same seed. Exploratory source searches
encountered two nonexistent guessed paths; these did not affect computation.
No evaluator assistance, human intervention or dependency installation occurred.

Successful script body: 6.38 seconds, including sampling/export/plotting but
excluding Python import/startup. Recovery sample-and-materialize: 1.74 seconds;
supplied-data sample-and-materialize: 1.24 seconds. Failed-run duration and total
agent wall time were not instrumented; consult harness timings rather than
interpreting these as end-to-end costs. Shared caches/load affect timings.
Active Python source totals 187 lines (18 model, 135 analysis, 34 checker),
plus the archived failed script. Token usage and authoritative tool-call counts
are harness-owned and not available as a reliable local measurement.

## Pending decisions

Human scientific review of assumptions, priors, predictive adequacy and recovery
noncoverage is still required. No scientific acceptance is claimed. The beta-prior revision is now complete as documented above; the separate
fresh-session handoff remains pending.
