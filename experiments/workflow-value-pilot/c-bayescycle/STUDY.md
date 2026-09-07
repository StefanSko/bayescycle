# Arm C — scripted revised proposal (current)

**Status: scripted prior revision completed; awaits human scientific review.**
This was a fixed evaluator request, not human scientific approval. No model or
result is human-approved. The separate fresh-session handoff remains pending.

## Current model and supersession map

Current executable: **`model_revised.py`**, with **beta ~ Normal(0, 0.25)**
(scale is SD). Only beta's prior changed; the module description marks its status.
All other priors, non-centered likelihood, design, estimand and assumptions below
are unchanged. `model.py` is the exact preserved initial executable, not the
current proposal. `STUDY-initial.md` preserves the complete pre-revision report.

| Current proposed artifacts | Superseded for current-model inference (preserved) |
|---|---|
| `results/revised-prior/` | `results/prior/` |
| `results/revised-recovery/` | `results/recovery/` |
| `results/revised/` | `results/initial/` |

All old model-dependent draws, predictions, recovery coverage, summaries,
diagnostics, trace/other figures and exported `fit.nc` are historical evidence,
not checks of the revised prior. Their bytes and summaries were not rewritten.
The original failed-prior preparation, logs, scripts, inputs and provenance are
also retained. Initial work was complete and its full recorded hash inventory
verified before revision; no baseline repair was performed.

## Revised execution and numerical results

Order: CLI prior prediction, CLI recovery plus supplementary summary, CLI
supplied-data fit plus supplementary checks, CLI trace export/plot. Every
inference command explicitly used `--backend bayesjax`; no backend fallback,
fit retry, dependency/source/shared-input edit, installation or retuning occurred.

Unchanged settings: float64, four chains, 500 warmup, 1000 retained draws per
chain, target acceptance 0.9, maximum tree depth 10. Seeds: prior 4200 (200 draws),
recovery 4301, supplied fit 4201, posterior prediction 4202 (200 draws).

| beta statistic | Revised supplied data | Revised recovery |
|---|---:|---:|
| Mean | 0.56116187 | 0.26166023 |
| SD | 0.05638898 | 0.05374478 |
| Equal-tailed 95% interval | [0.44983337, 0.67322471] | [0.15527495, 0.36553559] |
| MCSE(mean) | 0.00111458 | 0.00102619 |
| Rank Rhat | 1.00196835 | 1.00040028 |
| Bulk ESS | 2562.21 | 2751.83 |
| Tail ESS | 2601.64 | 2693.82 |
| Divergences | 0 | 0 |

No parameter coordinate crossed rank Rhat > 1.01 or bulk/tail ESS < 400 in either
revised fit. Independent diagnostics use ArviZ 0.22.0; native classic diagnostics
remain in NDJSON and are not substituted for rank diagnostics. Supplied-data beta
mean decreased by 0.02808066 from the initial result. This is a prior-sensitivity
comparison, not validation of the tighter prior.

**Recovery limitation persists:** beta truth 0.4 exceeds the revised interval
upper bound 0.36553559 (initial upper bound 0.38011747). All other parameter truth
coordinates remain inside their marginal intervals. One fixture is not a
calibration test; no primary-data generating truth was accessed.

Required current files are `results/revised/posterior.npz` and
`results/revised/result.json`, with the same schema as initial. Native CLI run,
IR, data, NDJSON, diagnostics, `fit.nc` and plots are retained alongside them.
Scalar arrays are (4,1000), z is (4,1000,8). Recovery and prior outputs have their
own directories. `analysis_revised.py` is a revision-only copy of the original
supplementary converter/checker with revised paths and source attribution; it
uses the same ArviZ definitions and seed-4202 conditional Normal posterior
prediction. These supplementary operations are not represented as CLI features.
The previously documented Bayesjax diagnose/PPC command limitations remain;
no incompatible diagnose/PPC plan was executed during revision.

## Revised figures actually inspected by this agent

Opened via the image tool, not human-reviewed:

- `results/revised-prior/prior_predictive.png`: broad mean/outcome spread;
  observed mean and SD lie within the simulated range. Arbitrary units prevent
  judging scientific plausibility without human context.
- `results/revised-recovery/trace.png`: overlapping densities and moving chains;
  tau remains right-tailed. Compact z overlays and some labels limit readability.
- `results/revised/cli-trace.png`: all coordinates have moving, overlapping chains
  without obvious separation; tau has occasional upper-tail excursions. Layout
  labels overlap. Supplementary `trace.png` saved but not separately opened.
- `results/revised/posterior_predictive.png`: observed mean/SD are central among
  replicates and marginal distributions broadly overlap. These marginal checks
  do not establish clinic-specific, residual, tail or structural adequacy.

## Revision provenance, commands, friction and pending decisions

`preservation-before-revision.json` inventories pre-existing evidence (including
initial source, all results/logs and original inputs). `verify_revision.py`
checks preservation; only beta's scale changes in both IR representations;
canonical data and native sampler settings match their initial counterparts;
source/data hashes and NDJSON fingerprints match; NPZ agrees with native draws;
result schema keys match initial; beta summaries reconstruct without refitting.
`revision-provenance.json` records separate exact-byte revised hashes and versions.
Original `provenance.json` is unchanged. Native absolute source paths still pose
the same relocation caveat as initial artifacts.

Reproduce revision only in a copy retaining initial artifacts but with no revised
outputs or `logs/revised/` (never delete completed evidence to force a rerun):

```sh
bash run_revised.sh
uv run --project .. python verify_revision.py --record
```

For a fresh agent inspecting completed work, **do not refit**:

```sh
uv run --project .. python verify_revision.py
```

Run commands/timing and outputs are in `run_revised.sh` and `logs/revised/`.
There were no failed revision commands. CLI trace export/plot succeeded with a
warning that a Bayesite binary was absent; it did not run that inference backend.
Wall times: prior 0.745 s, recovery 6.208 s, supplied fit 6.170 s, CLI plot/export
1.629 s. These exclude supplementary analysis and agent work and are cache/load
dependent. No new install or human intervention occurred. Overall revision took
approximately five minutes; exact token/tool totals are left to harness records.
The scripted evaluator request is recorded as such, not human scientific review.

Human decisions still required: assess the tighter prior's substantive basis,
review persistent recovery noncoverage, assess shared linear slope/exchangeable
clinic/Gaussian residual assumptions, decide whether richer checks or further
revisions are warranted, and decide whether any scientific use is appropriate.
The estimand remains an associational within-clinic contrast, not a causal effect.
Computational screens and marginal PPCs do not settle these decisions.

---

# Historical initial-stage report (superseded, retained verbatim below)

**Historical status at initial completion: initial stage completed; awaits human
scientific review.** The remaining sections describe the initial model/results
only, including then-pending work. Use the current revision sections above for
current status. The exact unedited original report is `STUDY-initial.md`.

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
