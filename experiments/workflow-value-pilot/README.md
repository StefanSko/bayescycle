# Does the custom Bayesian stack earn its complexity?

**[Open the interactive walkthrough →](report.html)** — one self-contained,
offline HTML file with the experiment timeline, result/prior comparisons, effort
filters, handoff checks, 18 original plots and 70 embedded evidence documents.
Open it directly in a browser; no server or installation is needed.

One small engineering pilot, run on 2026-09-04. **This is evidence for a smaller
prototype, not a verdict that the existing project should be deleted.**

## What was actually tested

Three independent Pi sessions implemented the same fixed hierarchical Gaussian
study, using the same model, data bytes, float64, 4 chains, 500 warmup and 1000
retained draws per chain. Each then received a scripted prior revision in a fresh
session, followed by a separate file-only handoff session. Every session used
`openai-codex/gpt-6-astra`, medium reasoning, with no unrelated extensions or
skills. Implementation sessions ran concurrently on the same machine.

- **A:** NumPyro + ArviZ, plain Python scripts.
- **B:** Bayeswire + direct public Bayesjax APIs + ArviZ.
- **C:** Bayeswire + Bayescycle CLI with the Bayesjax backend, native run artifacts,
  and the existing isolated CLI visualization integration. Unsupported
  diagnostics/prediction steps required supplementary Python scripts.

The [protocol](PROTOCOL.md) and [initial brief](INITIAL-BRIEF.md) were written
before analysis agents started. Generating truth for the main dataset was kept
outside their permitted workspaces. This was instruction-based separation, not
an OS sandbox. No agent needed evaluator assistance to repair its implementation.
No package source or root configuration was changed; pre-existing `examples/`
files were left alone. Dependencies are in a separate experiment lock/environment.

**Important reduction from the conversational proposal:** this unattended pilot
could not honestly test real human approvals. It used a fixed evaluator fixture,
not an agent proposing a model for you to approve. C tests the CLI/artifact layer,
not execution of the full phase-gated study skill. Human understanding, trust,
scientific decision quality, and enforceable approval gates remain **unmeasured**.

## Numerical results

Estimand: within-clinic expected outcome difference for a one-unit predictor
change, beta. This is associational, not an identified causal effect.

| | A: NumPyro | B: direct Bayesjax | C: Bayescycle CLI |
|---|---:|---:|---:|
| Initial beta mean | 0.588040 | 0.589243 | 0.589243 |
| Initial equal-tailed 95% interval | [0.47294, 0.69843] | [0.47530, 0.70470] | [0.47530, 0.70470] |
| Revised beta mean | 0.561321 | 0.561162 | 0.561162 |
| Revised equal-tailed 95% interval | [0.44791, 0.66930] | [0.44983, 0.67322] | [0.44983, 0.67322] |
| Initial implementation-session wall time | 3m 45s | 5m 02s | 7m 05s |
| Initial agent tool calls | 19 | 35 | 51 |
| Initial authored Python lines | 184 | 187 | 194 |
| Initial + revision + handoff session time | 9m 05s | 10m 19s | 13m 25s |
| Preserved original sources/results | Yes | Yes | Yes |
| Fresh-session numerical handoff | Pass | Pass | Pass |

C additionally authored a 25-line shell entrypoint. Python line counts include
arm-specific checking/provenance helpers, exclude dependencies, and are not a
maintainability score. Session wall time includes implementation, reading,
execution and reporting; it is **not a sampler performance benchmark**.

All initial and revised supplied-data means are within the predeclared
`4 * hypot(MCSE_A, MCSE_B)` screen. B and C's initial and revised posterior arrays
are bit-identical for every parameter, as expected from the same engine/settings.
Independent checks recompute every scalar parameter's rank Rhat, bulk/tail ESS,
beta summaries, array shapes, float64, positive scales and native divergences.
All 12 saved fits (initial, recovery, revised, revised recovery across three arms)
pass the specified thresholds, with zero retained-draw divergences. Saved
numerical summaries agree with recomputation. All nine agent sessions completed
within their deadlines. The extra failed-export recovery attempt in B is preserved
separately and is not counted as an additional completed fit in this total.

**Recovery caveat:** all three initial recovery intervals miss the known beta=0.4;
the tighter-prior revision does not fix that. Every analysis agent explicitly
reported the miss rather than tuning or hiding it. One recovery dataset cannot
establish calibration, and agreement between implementations is not independent
proof of scientific validity. The same held-out primary truth lies within all
reported main-study intervals; that is also descriptive only.

## What the friction actually was

- **A:** initial implementation completed without a failed tool call. Its revision
  encountered one non-unique text-edit match, then repaired the edit before running.
  This was a harness-editing error, not a NumPyro failure.
- **B:** the agent incorrectly assumed a native diagnostic object was a dataclass.
  Diagnostic export failed after recovery sampling. It preserved the failed
  attempt, fixed its adapter code and reran with the same seed/settings. The
  repeated recovery draws were identical; no favorable-seed search occurred.
- **C:** prior simulation rejected an observed `y` supplied with design inputs.
  The agent preserved the failed preparation and explicitly produced design-only
  input. `diagnose` and `posterior-predictive` dry plans selected Bayesite, not
  Bayesjax; it did not execute an unapproved backend switch. Instead it wrote
  ArviZ diagnostic and conditional predictive helpers. CLI export/trace plotting
  succeeded through pinned uvx environments; the first plot/export command took
  approximately 32 seconds. Source-navigation mistakes also contributed overhead.

The directly exposed Bayesjax simulation API currently exports prior prediction,
not a generic posterior-predictive helper. Both B and C duplicated the model's
forward equation in their predictive scripts; A used NumPyro's `Predictive`.
That is a concrete library-level usability gap this task exposed, not a reason
to add another orchestration system.

These are observations from one agent run per setup. They are not intrinsic
error rates, and the agent's familiarity with NumPyro is a possible confound.
Shared cache state and concurrent machine load confound speed comparisons.

## Revision, handoff, and provenance

All three revisions changed only beta's prior SD from 1 to 0.25, aside from model
docstrings. They retained the original executable model and completed initial
artifacts, wrote separate revised prior/recovery/fit outputs, and explained
supersession in `STUDY.md`. The evaluator independently snapshotted original
sources and input/result hashes before revision and checks preservation.

Fresh-session handoff results are in each arm's `HANDOFF.md` and
`handoff-result.json`; [evaluation.json](evaluator/evaluation.json) independently
checks their numerical reconstructions. **All three handoffs succeeded:** each
identified prior SD 0.25, recomputed the current numerical summary from draws,
checked available provenance, and correctly reported no human approval. None
refitted. Thus this test found no unique continuity advantage for C, although its
native artifacts provided additional standardized facts to verify.

The meaningful distinction is not simply “files vs no files”: all arms saved
files. C supplies standardized IR, canonical inputs and model/data fingerprints
out of the box. A and B built local source/input manifests in their analysis
scripts. The native C fingerprints and native-to-NPZ conversions independently
check out. Local hashes are not signatures or proof that arbitrary code ran as
claimed; C's input fingerprint also does not cryptographically authenticate every
posterior draw. No hostile-tampering or exact relocated-replay test was performed.

## Interpretation

1. **Existing tools delivered the numerical analysis without the custom stack.**
   A was the lowest-friction initial implementation on this fixture.
2. **The custom library worked correctly on the tested comparison, but did not
   demonstrate a usability advantage here.** Its different model language may
   still have value for validation, inspectability or other models not exercised.
3. **The CLI provided real provenance structure, but introduced additional steps
   and did not eliminate analysis glue for the Bayesjax workflow.** Ordinary
   scripts also preserved results and passed handoff. That supports questioning
   how much orchestration is necessary, not discarding provenance.
4. **The actual product hypothesis—better human-controlled Bayesian work—remains
   open.** An unattended, prescribed-model test cannot decide it.

A useful next move is one real human-led analysis with a small library/harness
integration, preserving model snapshots, results and decisions. Retain only
mechanisms that make a difference during that interaction. Do not rewrite the
repository or build an MCP server on the strength of this single pilot.

Tau and PydanticAI were not experimental arms. Tau's current
[Python extension API](https://twotimespi.dev/guides/extensions/) is a plausible
way to call a scientific library directly while reusing a terminal interface;
PydanticAI is another Python-native agent framework. In-process scheduling,
cancellation, state lifetime, and user interaction would need their own small
integration trial. This pilot held Pi fixed deliberately.

## Rebuild or check the HTML walkthrough

The HTML is a saved-evidence presentation, not another inference run. Its builder
uses only the standard library and reads existing documents/figures without
modifying them. The browser smoke test uses the root workspace's Playwright.

```bash
uv run python experiments/workflow-value-pilot/build_report.py
uv run python experiments/workflow-value-pilot/check_report.py
```

`report.template.html` contains the presentation. `report.html` embeds the data
and original PNGs, so it remains usable when copied away from this repository.
The offline browser test covers all four fit views, keyboard navigation, effort
filters, every embedded figure, document inspection, exact JSON downloads and a
narrow viewport. Screenshots are saved locally under `evaluator/html-*.png`.

## Inspect or reproduce

- `a-numpyro/`, `b-bayesjax/`, `c-bayescycle/`: actual agent-authored scripts,
  `STUDY.md`, input fixtures and result summaries.
- `evaluator/evaluation.json`: independently computed metrics and session usage.
- `evaluator/<arm>/<stage>/execution.json`: measured session timing/status.
- `evaluator/<arm>/initial-source/`: evaluator snapshots of initial Python source.
- `evaluator/input-manifest.json`, `primary-truth.json`: fixture hashes, versions,
  and held-out main-data truth. Analysis agents were forbidden to read these.
- Raw event streams, session transcripts, posterior draws and plots are retained
  locally but gitignored to avoid accidentally committing bulky evidence.

```bash
uv sync --project experiments/workflow-value-pilot
uv run --project experiments/workflow-value-pilot python \
  experiments/workflow-value-pilot/evaluate.py evaluate
```

The evaluation command requires the locally retained NPZ/native artifacts. It
recomputes summaries, does not refit, and updates only evaluator reports.
For a fresh full run, use a new clean experiment copy/worktree and follow each
arm's `STUDY.md`; analysis entrypoints deliberately refuse completed outputs.
`prepare.py` also refuses to regenerate an existing fixture manifest.
The bounded `run_agent.py` is a one-off process/log wrapper, not a new workflow
framework. Its prompts, model/provider and deadlines are explicit in this folder.

Validation: root workspace guards passed (14 tests). Evaluator helper scripts
pass Ruff checks/formatting. Scientific numerical checks are recorded separately
in `evaluation.json`. Agent-generated source/snapshots were not reformatted after
execution, which would invalidate the recorded source hashes.
