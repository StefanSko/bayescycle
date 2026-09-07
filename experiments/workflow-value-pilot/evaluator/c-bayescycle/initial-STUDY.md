# Arm C — initial engineering fixture

**Status: initial stage completed; awaits human scientific review.** This is the
fixed evaluator fixture, not a human-approved scientific study. No approval gates
were invoked or claimed. Revision and fresh-session handoff have NOT been run.

## Question and fixed model

Estimate beta: the expected within-clinic outcome difference for x → x+1.
This is an associational conditional contrast, not an identified causal effect.
There are eight clinics, 20 observations each, no missingness, and arbitrary
standardized units. x and the one-hot clinic design are treated as fixed.

`model.py` declares the model in Bayeswire:

- alpha ~ Normal(0, 2), beta ~ Normal(0, 1)
- tau ~ HalfNormal(1), sigma ~ HalfNormal(1), both constrained positive
- eight independent z ~ Normal(0, 1)
- y ~ Normal(alpha + tau * (clinic_design @ z) + beta*x, sigma)

Assumptions include a shared linear slope, exchangeable Gaussian clinic offsets,
conditionally independent Gaussian errors and common residual scale. The fixture
does not address confounding, predictor error, selection or generalization to new
clinics. Those assumptions require scientific review, not just sampler checks.

## Execution and results

Order: CLI prior prediction → separate CLI recovery fit and manual recovery
summary → supplied-data CLI fit. There was exactly one successful sampling run
per dataset, with no fit retries or setting changes. Every fitting invocation
explicitly used `--backend bayesjax`; no Rust engine was run.

Both fits used float64, 4 chains, 500 warmup, 1000 retained draws per chain,
target acceptance 0.9 and maximum tree depth 10. All requested settings were
available. Seeds: prior 4200 (200 draws), recovery 4301, initial 4201,
posterior prediction 4202 (200 draws).

| beta statistic | Initial | Recovery |
|---|---:|---:|
| Mean | 0.58924253 | 0.27318238 |
| SD | 0.05834072 | 0.05383361 |
| Equal-tailed 95% interval | [0.47530003, 0.70470422] | [0.16870437, 0.38011747] |
| MCSE(mean) | 0.00105029 | 0.00095978 |
| Rank Rhat | 0.99996346 | 1.00102898 |
| Bulk ESS | 3100.31 | 3167.74 |
| Tail ESS | 2454.99 | 2437.01 |
| Divergences | 0 | 0 |

ArviZ **0.22.0** independently computed rank-normalized Rhat, bulk/tail ESS and
MCSE from saved chain-by-draw samples. No scalar coordinate in either fit crossed
Rhat > 1.01 or ESS < 400. These are computational screens, not proof of model
correctness. Native Bayesjax classic split Rhat and ESS are preserved in the
NDJSON trailer; they are not interchangeable with the ArviZ diagnostics.

**Recovery caveat:** beta truth 0.4 lies above its 95% interval. Alpha, tau, sigma
and all eight z truth values are inside their marginal intervals. This one
recovery dataset is descriptive only, neither a calibration pass nor a reason to
hunt for another seed. Primary-data generating truth was not accessed.

## Artifacts and supplementary operations

- `results/prior/`: native CLI run, prior NDJSON, `prior_predictive.npz`, figure.
- `results/recovery/`: native CLI run, `posterior.npz`, `sample_stats.npz`,
  `result.json` including every parameter and truth coverage, trace figure.
- `results/initial/`: native CLI run, same posterior/statistic/summary files,
  `posterior_predictive.npz`, predictive and trace figures; CLI-exported
  `fit.nc` and `cli-trace.png` also retained.
- `logs/`: command output, timing, diagnostic/PPC dry plans and execution errors.
- `provenance.json`: exact-byte SHA-256 inventory, dependency versions, device
  and environment facts, and successful source/canonical-data consistency checks.

Scalar posterior arrays are (4,1000); z is (4,1000,8). Predictive y is (200,160).
`analysis.py` is explicitly supplementary, not a CLI feature: it validates the
posterior format and fingerprint, converts native NDJSON, computes independent
ArviZ diagnostics, performs recovery truth checks, plots predictive checks and
an additional trace, and simulates conditional Normal posterior predictions.
Posterior prediction selects 200 distinct posterior indices with NumPy RNG seed
4202, then samples residual noise at the unchanged observed clinic design.
Indices and conditional means are saved for inspection.

The adapter does not serve dedicated recovery/check commands for Bayesjax, so
recovery used `bayescycle sample` on the supplied recovery fixture. Dry plans for
`diagnose` and `posterior-predictive` selected **bayesite**, despite a Bayesjax
source run. These plans were NOT executed; their missing Bayesjax support was
handled by the supplementary script, not backend fallback.

CLI trace plotting succeeded through its pinned standalone uvx export/plot
boundary, taking **32.001 s** including export/environment overhead. Its log
warned that a Bayesite executable was absent; export and plotting nevertheless
succeeded without a Rust engine. Independent ArviZ summaries use the shared
0.22 environment, not the visualization environment's diagnostic definitions.

## Figures actually inspected by this agent

Opened with the image read tool:

1. `results/prior/prior_predictive.png`: broad prior outcome and mean ranges;
   observed mean and SD fall inside the simulated spread. Scientific plausibility
   cannot be settled because units are arbitrary and no human has reviewed it.
2. `results/recovery/trace.png`: overlapping chain densities and moving traces;
   tau has a long right tail. This compact plot overlays z coordinates.
3. `results/initial/cli-trace.png`: all coordinates show overlapping moving
   chains, no obvious chain separation; tau has occasional upper-tail excursions.
   Some axis labels overlap in the CLI layout.
4. `results/initial/posterior_predictive.png`: observed mean and SD are central
   among replicates and the marginal distribution broadly overlaps. This is a
   limited marginal check, not an exhaustive clinic/residual/tail assessment.

The supplementary initial `trace.png` was saved but not separately opened.
No human inspection is implied by any of these statements.

## Provenance and reproducibility

Original input files remain unchanged. CLI `run.json` records source hashes,
backend and settings. Native model/data fingerprints were recomputed from exact
received `model.ir.json` and canonical `data.json` bytes and matched both posterior
header and trailer. Canonical fit arrays were independently checked against the
original input arrays, including float64 and one-hot clinic counts. Initial,
recovery and prior model IR bytes are identical.

Original source SHA-256:

- `model.py`: `9a91bceb9a1b815e5b369c49814a2be57f030d2902b27a0750bb1715279bd169`
- `input/data.json`: `8a753a950786ff60f47747dd0176ff01619d2f958402e0d042299fcb4523018d`
- `input/recovery.json`: `edd93d9672190fe63b29312c58faa7e8f03dd70c0220dade8ed22177753c281b`
- `input/recovery-truth.json`: `b1f02352fcc4565268a145ac8239ec5f9ffa7ced999f45652c26332d960715cf`

The full exact-byte inventory is in `provenance.json`. Original byte hashes are
not replaced with hashes of reserialized canonical data. Native run metadata uses
absolute source paths, so CLI replay after relocation may require path handling;
local native artifacts and this relative-path inventory remain inspectable.

Run from a **fresh workspace copy with no outputs**, within the supplied locked
project environment:

```sh
bash run_initial.sh
```

The shell script refuses any existing `results`, `design.json` or provenance
inventory. It documents the corrected successful command sequence; the original
execution used those commands individually. `analysis.py` refuses existing
posterior/result outputs and uses exclusive creation for derived NPZ/JSON.
Do not delete completed results to force a rerun. The original run is preserved.

For individual postprocessing on a fresh native run (not completed derivatives):

```sh
uv run --project .. python analysis.py prior
uv run --project .. python analysis.py recovery
uv run --project .. python analysis.py initial
uv run --project .. python record_provenance.py
```

## Friction, timing and remaining work

- Initial prior command passed y as well as design; CLI rejected `Unexpected
  model data: ['y']` (exit 2), after writing preparation artifacts but before
  simulation. Retained under `results/failed-prior-with-y/`, with `logs/prior.log`.
  `prepare_design.py` explicitly extracts x/design into a new file. The corrected
  prior run used the same prescribed seed, not a fit retry.
- One attempted public-source read and one search used obsolete source/test
  paths and failed. No private API was used and no package code was modified.
- Successful CLI recovery fit: 6.346 s wall; initial fit: 6.241 s wall.
  Shared installation time is excluded. Overall agent implementation/execution
  took approximately 8 minutes; timing is approximate under shared machine load.
- Authored executable code: 219 lines across five files. No evaluator assistance
  or human intervention occurred. Exact LLM token usage and harness tool-call
  totals are not available from these local artifacts; the harness must supply
  them rather than this report inventing counts.
- Pending: human scientific review, assessment of recovery noncoverage, richer
  model checks as appropriate, and separately requested revision/handoff stages.
  No scientific acceptance, causal identification or broad workflow superiority
  follows from this single engineering fixture.
