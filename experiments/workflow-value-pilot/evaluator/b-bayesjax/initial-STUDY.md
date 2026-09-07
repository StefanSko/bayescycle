# Arm B — initial engineering fixture

**Status: computational initial stage complete; awaits human scientific review.**
This is an evaluator-fixed engineering test, not a human-approved scientific
study. No approval gates were invoked or satisfied. No revision or fresh-session
handoff stage has been executed. No Bayescycle CLI or internals were used.

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
noncoverage is still required. No scientific acceptance is claimed. The requested
later beta-prior revision and fresh-session handoff remain unexecuted.
