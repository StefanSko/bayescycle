# Workflow value pilot — registered before running the analysis agents

## Question

Does the custom Bayesian model library improve on NumPyro for an agent conducting
an inspectable analysis? Does the CLI/artifact layer improve on direct library use?
This is one deliberately small, non-blinded engineering pilot, not a scientific
calibration study or a decisive product benchmark.

## Arms

- A: NumPyro + ArviZ, ordinary Python scripts.
- B: Bayeswire + direct public Bayesjax APIs + ArviZ, ordinary Python scripts.
- C: Bayeswire + Bayescycle CLI, explicitly `--backend bayesjax` for sampling.
  Use the existing artifact contract and public CLI where supported. Any extra
  scripts needed for missing operations must be identified rather than disguised
  as CLI features. No silent Rust-backend fallback.

All arms share a separately locked Python environment, JAX/BlackJAX versions,
input bytes, model specification, and sampler settings. NumPyro's NUTS and
BlackJAX NUTS are different implementations; identical seeds do not imply
identical draws. No library source or repository configuration changes allowed.

## Important scope adjustment

The full existing study skill is phase-gated and requires actual human decisions.
It cannot honestly be evaluated by having an unattended agent approve itself.
This first run therefore tests computational execution, scripted revision,
and file-based handoff. C tests the CLI/artifact layer, NOT successful completion
of the entire study skill. The fixed model/settings below are an evaluator fixture,
not invented human approvals. Human understanding, subjective trust, and genuine
approval-gate behavior remain UNMEASURED. No agent may claim scientific approval.

## Shared scientific fixture

Eight clinics, 20 observations per clinic, a continuous predictor x and continuous
y. All arrays are supplied in `input/data.json`; `clinic_design` is the N x 8
one-hot design matrix. Outcome and predictor units are arbitrary standardized units.
No missingness; x is treated as fixed. Model (non-centered):

- alpha ~ Normal(0, 2)
- beta ~ Normal(0, 1)
- tau ~ HalfNormal(1)
- z_j ~ Normal(0, 1), j = 0,...,7
- sigma ~ HalfNormal(1)
- y_i ~ Normal(alpha + tau * (clinic_design @ z)_i + beta*x_i, sigma)

Estimand: within-clinic expected outcome difference for x -> x+1, namely beta.
This is an associational conditional contrast, not an identified causal effect.

Use float64, 4 chains, 500 warmup and 1000 retained draws per chain,
target acceptance 0.9, maximum tree depth 10. Seed 4201 for supplied-data fit.
No automatic tuning changes; report diagnostic failures rather than conceal them.
Use 200 prior predictive draws (seed 4200); 200 posterior predictive draws
(seed 4202). If a public backend does not expose a setting, document it.

One separate recovery fixture is supplied in `input/recovery.json` with truth in
`input/recovery-truth.json`. Use seed 4301. The generating truth for the primary
dataset is held outside agent workspaces. Agents must not read it. Truth coverage
on one dataset is descriptive only, not a calibration pass/fail test.

## Required initial outputs

- An executable script with clear commands and immutable result directories.
- Saved chain-by-draw posterior samples (NPZ with named parameters is acceptable;
  C must also preserve native run artifacts).
- `result.json`: estimand mean, SD, equal-tailed 95% interval, MCSE(mean), rank Rhat,
  bulk/tail ESS, divergence count, sampler settings, paths to samples and figures.
- Prior predictive, trace, and posterior predictive figures; describe what the
  agent actually inspected, not what a human supposedly reviewed.
- `STUDY.md`: question, assumptions, known limitations, completed work, explicit
  statement that outputs await human review, and reproducible commands.
- Recovery and initial-fit artifacts must be distinct.

ArviZ 0.22 is supplied for a common independent diagnostic definition in all arms.
Do not compare Bayesjax classic split Rhat to rank-normalized Rhat as if identical.

## Revision (requested only after initial outputs)

Change ONLY beta's prior to Normal(0, 0.25). Preserve initial artifacts, identify
which model-dependent checks/results are superseded, rerun relevant checks and
fit using seed 4201, and clearly distinguish initial vs revised results. Store
revised samples and summaries separately. Remain awaiting human review.

## Fresh-session handoff

A new agent sees only the arm's files and general instructions, not earlier
conversation, evaluator truth, or other arms. It must identify the current model,
reproduce beta's numerical summary from saved draws without refitting, verify the
model/data provenance to the extent artifacts permit, and state pending decisions.
This tests summary reconstruction, not exact whole-analysis replay after relocation.

## Evaluation and interpretation (fixed before fits)

- Computation: independently recompute diagnostics from saved draws. Flag rank
  Rhat > 1.01, bulk or tail ESS < 400, or nonzero divergences. Failures are evidence,
  not authorization to silently change the fixture.
- Compare beta means between arms with tolerance 4*sqrt(mcse_A^2+mcse_B^2).
  This is a conservative pilot screen, not a formal equivalence test. Do not treat
  posterior agreement as independent proof of model correctness.
- Inspect model implementations for equivalence to the fixture.
- Record wall time, LLM usage, tool invocations, failed tool invocations, manual
  intervention, and code size. Shared installation time is separate. Wall-time
  comparisons are approximate due to shared caches and machine load.
- Record revision preservation, truthful supersession, and handoff success.
- Subjective human understanding/control are unscored.

Budget: up to 15 minutes per initial implementation, 10 per revision, 5 per
handoff. Timeouts count as incomplete; retain partial evidence. One attempt per
arm; ordinary self-repairs within budget count as friction. Evaluator assistance
must be recorded. No custom MCP server or new general benchmark framework.

If A is comparable to B, the custom library has not demonstrated value on this
fixture. If B improves on A but C does not improve on B, retain the library and
question orchestration. If C uniquely protects provenance or continuity, identify
that mechanism specifically. One fixture cannot justify deleting a project.
