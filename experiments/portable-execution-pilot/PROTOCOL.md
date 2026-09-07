# Portable execution and predictive model reuse — pilot protocol

**Status: draft; not yet frozen for execution.** No new inference, agent trial,
engine installation, or product implementation is recorded by this document.
Resolve the pre-run gates below before running the experiment. Preparation and
execution are separate activities; missing capabilities must not be implemented
silently to make this baseline pass.

This follows the [retrospective and next-test discussion](../agentic-bayesian-vision-and-next-tests.md).
The [hosted pilot](../workflow-value-pilot/README.md) and
[local-agent pilot](../gemma-workflow-pilot/README.md) remain unchanged.

## 1. Question and bounded claim

Can a recipient execute prior prediction, sampling, diagnostics, and posterior
prediction for two fixed models using only a prebuilt Bayesite executable and
model/data documents, offline, without reimplementing the model equations?

This is a product-capability pilot, not an agent benchmark or sampler speed
comparison. A pass supports the claim only for the pinned binary, target, models,
and commands tested. It does not overturn NumPyro's lower observed authoring
friction in the earlier pilot.

**Included:** direct native Bayesite CLI, machine-readable artifacts, same-design
prediction for existing observations/groups, and independent numerical checks.

**Excluded:** Python/eDSL authoring on the receiver; Bayescycle CLI; plotting,
NetCDF and `uvx`; browser/WASM; recovery/SBC; new-group prediction; agent repair;
human approval quality; full relocated replay; revision-integrity guarantees;
security audit; and cross-platform or bitwise reproducibility claims.

The receiver may use the declared OS/kernel, container runtime, and host-side
measurement tools. These are infrastructure, not secretly free dependencies.
“One binary” here means one scientific runtime executable, not literally an
executable without an operating system. Resource limits contain this trial;
they do not establish architectural safety.

## 2. Pre-run gates and frozen record

Before the first scientific command, create `evaluator/frozen-manifest.json`,
record its digest in `evaluator/events.ndjson`, and retain the exact protocol and
validator source it identifies. The append-only event log is an audit aid, not
an authenticated attestation. All entries below must be resolved:

- **Engine:** proposed release `v0.4.0`, target
  `x86_64-unknown-linux-musl`, matching the currently configured Bayescycle pin.
  Record release URL, archive/checksum-file digests, executable SHA-256 and byte
  size, and available version/capability metadata. Verify the published checksum.
  Do not substitute the unverified executable on local `PATH` or a stale build.
  A development binary requires an explicit pre-run amendment, source revision,
  clean/dirty status, build provenance, and a separately labelled dev-only claim.
- **Receiver:** select and record the native Linux x86_64 host/VM, kernel, CPU,
  memory, container runtime/version, image digest, CPU allocation and limits.
  Do not use transparent ARM-to-x86 emulation in the primary run. An unavailable
  suitable receiver is a blocked setup, not evidence that Bayesite failed.
- **Isolation:** use a scratch-based container image containing the executable
  only, with no Python, shell, scientific libraries or package manager. Inspect
  its filesystem and executable linkage before freezing. Use a read-only root,
  non-root user, dropped capabilities, no network, read-only fixture mount, and
  one writable output mount; no checkout, home directory, sockets, or caches.
  Declare bounded temporary storage if the executable requires it. Retain the
  exact image recipe, mount configuration, and enforcement evidence.
- **Inputs:** materialize and hash both fixtures and negative cases below.
  Authoring/serialization may use Python on the producer, never on the receiver.
  Record producer versions and source hashes. Inspect compatibility from schemas,
  capabilities and source; do not run exploratory fits or predictions first.
- **Evaluator:** freeze its source, independent reference-artifact hashes, and
  separate Python environment/lock, including NumPy and ArviZ versions. Use
  `uv run ...` for project-side preparation and validation commands.
- **Runner:** freeze exact argument vectors, working directories, environment
  allowlist, output paths, timeout handling, telemetry method and result schema.
  Infrastructure probes may inspect linkage/version/capabilities but must not
  sample, predict, tune the fixture, or repair product code.

Documentation inspected at drafting: Bayescycle checkout
`41df90b7c4e0904d19d03a53fa26c8fbe3d504ce`; adjacent Bayesite checkout
`1d8f1746fe1ca7ef11b25be31fd6df20c2faad0d` (clean at inspection). These are
inspection references, **not proof of the released binary's contents**.

If inspection shows a required operation/model is unsupported, record it as a
capability gap; do not replace the fixture with an easier model. Acquisition or
harness failures are recorded separately from engine failures. No outcome-driven
changes to settings, thresholds, model, or binary are allowed after freezing.

## 3. Fixed fixtures

### A — Analytic Gaussian location

- `theta ~ Normal(0, 1)`.
- Four fixed offsets `x = [0, 0, 0, 0]`.
- `y_i ~ Normal(theta + x_i, 1)`.
- Observations `y = [-1, 0, 1, 2]`.
- One scalar parameter; `x` and `y` have shape `[4]`, float64.

The offsets make the predictive vector shape explicit without supplying observed
outcomes to prior prediction. The exact posterior is
`theta | y ~ Normal(0.4, sqrt(0.2))`; its second moment is `0.36`.
The prior predictive marginal variance is `2`, with covariance `1` between
positions; posterior predictive marginal variance is `1.2`, with covariance
`0.2`. Preserve one canonical IR file for all operations.

### B — Original hierarchical Gaussian fixture

Reuse the **initial**, not revised, eight-clinic/160-observation model from the
hosted pilot:

- `alpha ~ Normal(0, 2)`, `beta ~ Normal(0, 1)`;
- `tau ~ HalfNormal(1)`, `z_j ~ Normal(0, 1)` for eight clinics;
- `sigma ~ HalfNormal(1)`;
- `y_i ~ Normal(alpha + tau * (clinic_design @ z)_i + beta*x_i, sigma)`.

Copy these retained artifacts byte-for-byte; paths are relative to
`experiments/workflow-value-pilot/`:

| Destination | Existing source | SHA-256 |
|---|---|---|
| `fixtures/clinic/model.ir.json` | `c-bayescycle/results/initial/model.ir.json` | `2da7c00eadf034cea33c7f10f0d98510233b1074653f619de9bd711097763f65` |
| `fixtures/clinic/data.json` | `c-bayescycle/results/initial/data.json` | `3231a98c65616c2edaff34e572ed9b59887b88a0ce1b2a16a0b7b725964abd34` |
| `fixtures/clinic/design.json` | `c-bayescycle/results/prior/data.json` | `6204ef1441b60ab12fcfe4da1002d5745cec70dfa685f36f260a4107fa48b7b9` |
| Evaluator only: NumPyro reference | `a-numpyro/results/initial/posterior.npz` | `1fa21d45fa47f2a9cc17f5d49ab1d044b80da447def796b95763c682ea7d3fba` |

These files are currently retained locally, not guaranteed to be available in a
fresh clone. Missing or mismatched evidence blocks preparation; do not recreate
it under the same identity. The canonical data hash differs from the earlier
raw-input manifest by representation. Independently check decoded values against
the original fixture, and verify that design contains precisely the declared
predictors without `y`. No generating truth is needed for this pilot.

For both fixtures, `data.json` contains observations; `design.json` omits them.
Posterior prediction uses the exact fit data, not the design-only document.

## 4. Receiver procedure and budgets

Run sequentially, analytic fixture then clinic fixture. Allocate four CPUs,
2 GiB memory and a 1 GiB output-size ceiling. Each sample command has a 15-minute
wall deadline; each other scientific command has two minutes. The host watchdog
must kill the full container on timeout and preserve partial output. One primary
attempt per command/fixture; no seed search, automatic retuning, or replacement
run. Independent operations may continue after failure; commands lacking a valid
prerequisite are recorded as `not_run`, never passed.

For each fixture, execute these argument templates as separate container
entrypoint invocations. `/input` mounts that fixture; `/output` is fresh for it.
No shell or model-aware helper runs inside the receiver.

```text
bayesite prior-predictive --model /input/model.ir.json --data /input/design.json
  --seed 4200 --draws 1000 --out /output/prior.ndjson
bayesite sample --model /input/model.ir.json --data /input/data.json
  --seed 4201 --chains 4 --warmup 500 --draws 1000
  --target-accept 0.9 --max-treedepth 10 --out /output/posterior.ndjson
bayesite diagnose --fit /output/posterior.ndjson
  --out /output/diagnostics.json
bayesite posterior-predictive --model /input/model.ir.json --data /input/data.json
  --fit /output/posterior.ndjson --seed 4202 --out /output/predictive.ndjson
```

Posterior prediction is expected to emit one replicate per retained posterior
draw (4,000 here), not the 200-replicate setting from the earlier agent pilot.
This is not a matched timing comparison to that pilot.

Record actual argv, environment, start/end time, exit/signal status, stdout,
stderr, peak memory, output sizes/hashes, and before/after input hashes. Capture
subprocess and network-attempt telemetry externally where available; distinguish
“network blocked” from “no connection attempted.” If tracing is unavailable,
say so rather than claiming an observed absence of attempted capabilities.
Record image/archive/executable/input delivery sizes and preparation separately.

For startup characterization, predeclare a version/capabilities command available
in the pinned binary. Record its first invocation and ten sequential repetitions
in fresh containers, including container-launch overhead. Report first-use and
warm-host results separately; do not call these filesystem-cold timings or infer
“millisecond engine startup” from measurements that cannot isolate engine time.
There is no binary-size or latency superiority threshold in this pilot.

## 5. Independent acceptance checks

Run these **after** receiving outputs, outside the receiver. Evaluator equations
are permitted for independent checking, never to manufacture missing predictions
or scientific outputs. Freeze their implementation before receiver execution.

### Artifacts and numerical sampling

- All four commands must finish within their budgets, exit zero, and produce
  complete, parseable artifacts with expected format markers, counts, shapes,
  parameter names, chain/draw lineage, and finite float64-interpreted values.
  Parsing JSON into float64 does not independently prove internal precision.
- Require unchanged model/input bytes and correct fit header/trailer model/data
  fingerprints, calculated from exact received bytes using the normative spec.
- Require positive `tau` and `sigma` in the clinic fit; exactly 4 x 1,000 retained
  draws for every parameter coordinate in both fits.
- Independently recompute rank-normalized Rhat, bulk/tail ESS, and divergence
  counts. Require Rhat <= 1.01, both ESS >= 400 for every scalar coordinate, and
  zero retained-draw divergences. Missing/undefined required diagnostics fail.
  Check native diagnostics under their documented definition; do not equate
  classic split Rhat with rank-normalized Rhat. Native report metadata and sampler
  counts must agree with the fit; freeze any native-formula numerical tolerances
  in the validator before execution.
- Analytic fit: means of `theta` and `theta**2` must be within five corresponding
  chain-aware MCSEs of `0.4` and `0.36`. MCSEs must be positive and finite.
- Clinic fit: compare beta mean with the frozen NumPyro draws, requiring
  `abs(mean_rust - mean_numpyro) <= 5 * hypot(mcse_rust, mcse_numpyro)`.
  Recompute the reference summary independently. Report all other parameter
  summaries as descriptive checks, without adding post-hoc acceptance tests.

### Prediction and model reuse

- Prior output must contain 1,000 joint parameter/outcome draws; posterior
  output must contain 4,000 outcomes linked to the correct source fit draws.
  Require outcome shapes `[4]` and `[160]`, respectively.
- On evaluator side only, compute each draw's conditional mean and scale from
  its parameters and frozen design. Standardize each replicated outcome as
  `r = (y_rep - mu) / sigma` (unit sigma for the analytic fixture).
- For each predictive stream, reduce each draw to two values: the mean of `r`
  across observations and the mean of `r**2`. Across draws, require those means
  to be within five standard errors of `0` and `1`, respectively. Use sample
  SD of the draw-level reductions divided by sqrt(draw count); require finite,
  positive SEs. These screens assume fresh independent conditional outcome
  noise; they do not use MCMC ESS for the residuals.
- For the analytic prior, additionally check the first and second moments of
  generated `theta` against `0` and `1`, with the same five-SE rule. Report
  predictive marginal variance/covariance against the analytic targets above
  as descriptive results, not extra acceptance gates.
- Review the entire receiver payload and invocation log: only the same IR and
  generic command plumbing may implement the workflow. Duplicated forward
  equations, Python adapters, engine switches, or evaluator-produced substitute
  artifacts fail model reuse even if their numerical results look correct.

These finite-sample screens can fail by chance and are not calibration or formal
equivalence tests. A failure is retained, not repaired by changing a seed or
increasing draws. Passing does not prove general inference/predictive correctness.

## 6. Negative cases

Prepare two separate copies of analytic inputs before freezing:

1. Change only the IR version field to `999`.
2. Keep valid IR but make the length-four `x` data document declare shape `[4]`
   with only three values; keep `y` unchanged.

Invoke `sample` on each in a fresh output directory, with the same settings and
a ten-second deadline. Both must terminate nonzero with one parseable JSON error
carrying the documented error marker and an explanation identifying the bad
version or shape/count, respectively. No successful fit may be emitted. Retain
any partial files. Crash, hang, malformed error, silent coercion, or fallback
fails the corresponding negative case. This tests two rejection paths, not
comprehensive parser safety or agent repair effectiveness.

## 7. Reporting, stopping, and follow-up

Keep a per-command and per-fixture result matrix. Separate:

- deployment/execution capability;
- numerical acceptance;
- model reuse;
- negative-case rejection;
- infrastructure blocks and unavailable measurements.

An **overall pass** requires every required positive and negative check on both
fixtures. A capability failure or numerical-screen failure is not an overall
pass; an unresolved setup/validator problem leaves the affected result blocked
or unassessed, not an inferred engine failure. Measurement gaps qualify the
performance/observability claims and must be listed explicitly.

Retain the protocol snapshot, frozen manifest, producer/validator sources and
lock, receiver recipe/configuration, input documents, exact commands, all logs,
raw outputs, hashes, telemetry, and independent `evaluator/evaluation.json`.
Produce a concise `README.md` with the result matrix and limitations, including
failures. Use exclusive/fresh output directories and never overwrite either the
old pilots or this pilot's completed evidence. A rerun after fixes is a new,
explicitly amended experiment, not a replacement primary result.

This pilot has no head-to-head NumPyro deployment arm and makes no relative speed
or size claim. Matching NumPyro's existing predictive reuse closes a gap on the
Rust path; it does not fix the Bayesjax legacy posterior-predictive API.

If this baseline passes, next test revision integrity and relocated replay as
separate guarantees, distinguishing self-contained generation runs from other
run profiles that still verify external source paths. Then run repeated,
budgeted agent authoring/repair trials against the capabilities actually shown.
If it fails, decide explicitly whether the missing capability merits product
work before commissioning another agent comparison.
