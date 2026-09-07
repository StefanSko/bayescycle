# Arm A — initial NumPyro engineering fixture

**Status: completed initial computational stage; awaits human scientific review.**
The model/settings are evaluator fixtures, not a human-approved scientific study.
No full gated study skill, scientific approvals, revision, or fresh-session handoff
were executed. No Bayeswire, Bayesjax, or Bayescycle was used.

## Question and fixed assumptions

Estimate the within-clinic expected outcome difference for x -> x+1: beta.
This is an associational conditional contrast, not an identified causal effect.
The supplied 160 rows cover eight clinics with 20 observations each; units are
arbitrary standardized units, no missingness, and x/design are treated as fixed.

`model.py` implements exactly:
- alpha ~ Normal(0, 2), beta ~ Normal(0, 1)
- tau ~ HalfNormal(1), eight independent z_j ~ Normal(0, 1)
- sigma ~ HalfNormal(1)
- y_i ~ Normal(alpha + tau*(clinic_design @ z)_i + beta*x_i, sigma).

Assumptions include a common linear slope, exchangeable Gaussian clinic
intercepts, homoskedastic Gaussian residuals, and conditional observation
independence. There is no random slope, measurement-error model, missingness
model, or causal identification argument. Eight groups provide limited
information about the population intercept scale.

## Execution and results

`analysis.py` executed in order: 200 prior predictive draws (seed 4200), separate
recovery fit (4301), then supplied-data fit (4201), then 200 posterior predictive
draws (4202). Both fits used float64, four parallel CPU chains, 500 warmup and
1000 retained draws per chain, target acceptance 0.9, maximum tree depth 10.
No settings were unavailable, no seed hunting, no retries or tuning changes.
Posterior prediction samples 200 of 4000 chain-major draws without replacement,
using split keys from seed 4202; saved indices identify the source draws.
Predictions are conditional on the existing clinics, not new-clinic predictions.

| Beta statistic | Supplied data | Recovery |
|---|---:|---:|
| Mean | 0.5880398931 | 0.2756737640 |
| SD (ddof=1) | 0.0576594246 | 0.0573885920 |
| Equal-tailed 95% interval | [0.4729374928, 0.6984312517] | [0.1624412186, 0.3889874405] |
| MCSE(mean) | 0.0009892939 | 0.0011201082 |
| Rank Rhat | 1.0012992240 | 1.0023250634 |
| Bulk ESS | 3395.7716 | 2631.4439 |
| Tail ESS | 2676.1942 | 2510.2828 |
| Divergences | 0 | 0 |

ArviZ 0.22 computed independent rank Rhat, bulk/tail ESS and MCSE on retained
chain-by-draw samples. No parameter (including individual z components) exceeded
Rhat 1.01 or had bulk/tail ESS below 400 in either fit. No tree-limit hits were
observed (maximum 95 leapfrog steps in both runs). These are computational
checks, not evidence of scientific validity. The diagnostics CSV includes
ArviZ's default 94% HDIs; the requested **95% equal-tailed** intervals are in
`result.json` and the table above, not those CSV HDI columns.

**Recovery limitation:** known beta=0.4 lies outside its 95% interval. All other
provided scalar/component truths lie inside their marginal 95% intervals.
This one-fixture coverage result is descriptive, neither a calibration pass/fail
test nor a reason to alter the fixed settings. Primary generating truth was not
read. More recovery fixtures would be needed to assess calibration, but were
not run under this initial-stage protocol.

## Figures actually inspected by this agent

The read tool displayed all four figures:
- `results/prior/prior_predictive.png`: broad prior distributions, some narrow
  and some very diffuse. Observed mean and SD lie within simulated ranges;
  the prior can generate much more extreme outcomes. This is not human prior approval.
- `results/initial/posterior_predictive.png`: observed outcome distribution broadly
  overlaps replicates; observed mean and SD are within the replicate distributions.
  No obvious gross discrepancy in these aggregate summaries. Residual patterns,
  clinic-specific misfit, and extrapolation are not comprehensively checked.
- `results/initial/trace.png` and `results/recovery/trace.png`: chain densities
  broadly overlap with no obvious persistent drift or stuck chains. Alpha,
  tau and z show more autocorrelation than beta; compact overlapping z traces
  limit individual visual inspection. Numerical diagnostics cover every component.

## Artifacts, provenance and reproduction

- `results/initial/` and `results/recovery/`: named-parameter `posterior.npz`,
  `sampler_stats.npz`, `result.json`, `diagnostics.csv`, `trace.png`, `COMPLETE`.
  Scalars have shape (4, 1000); z has shape (4, 1000, 8).
- `results/prior/prior_predictive.npz`: prior parameters and y, with 200 draws.
- `results/initial/posterior_predictive.npz`: y of shape (200, 160) and selected
  posterior indices. Native NumPyro divergences, acceptance probabilities,
  leapfrog counts and potential energies are retained in sampler_stats.
- `results/provenance.json`: SHA256 hashes of exact original input bytes,
  model and analysis source, brief and protocol; software versions, devices,
  environment and command. Inputs were unchanged. This is an inspectable local
  provenance manifest, not signed or externally tamper-proof storage.
- `execution.log` captures the run; `results/execution.json` records stage order,
  two fits and zero retries. `COMPLETE` markers distinguish completed artifacts.

From this directory, the original command is:

```sh
uv run --project .. python analysis.py
```

**It refuses any existing `results/` directory before writing artifacts**, even
partial attempts. Do not delete completed results to rerun. For future authorized
reproduction, use a separate clean copy of this arm under the same project
layout, retaining original input bytes and the shared brief/protocol paths.
No alternate destination was created or rerun during this task. Script-enforced
immutability prevents accidental reruns, not deliberate filesystem edits.

Exact SHA256 identifiers (full manifest also covers script and recovery truth):

```
model.py             87a1a39f12a98dd743b449ac1b788f780f1307ce08671e08e1e679d383a21f4d
input/data.json      8a753a950786ff60f47747dd0176ff01619d2f958402e0d042299fcb4523018d
input/recovery.json  edd93d9672190fe63b29312c58faa7e8f03dd70c0220dade8ed22177753c281b
```

## Engineering overhead and pending decisions

Analysis-body wall time was 3.459 seconds (excludes imports/process startup);
recovery fit 0.871 seconds and initial fit 0.835 seconds, with NumPyro arrays
materialized before timing ended. Shared compilation caches/machine load may
strongly affect these times; they are not isolated backend benchmarks.
Agent implementation and inspection took approximately three minutes starting
at the first recorded shell timestamp 20:36:07 UTC. Code size: 184 Python lines
(170 analysis, 14 model). No failed commands, dependency installation, manual
intervention, evaluator assistance, or scratch fit attempts occurred. Tool calls
and LLM token usage are available to the harness; token counts are not exposed
to this script and are not fabricated here.

Pending: human scientific review of question, assumptions, priors and checks;
interpretation of the recovery miss; any requested model revision; and the
separate fresh-session handoff. No current artifacts are superseded.
