# Does the narrower interface help the installed local agent?

**Result: this pilot did not support that hypothesis.** The full-study trials
never reached model authoring or inference. A smaller, exploratory authoring probe
succeeded with NumPyro but not with Bayeswire within its fixed budget. This is not
a claim that the custom inference code is incorrect or that Gemma can never use it.

Run on 2026-09-04. The hosted pilot and its HTML are preserved separately.

## Installation actually tested

- Existing `ollama/gemma4-pi:12b`: Gemma 4 11.9B, Q4_K_M; Ollama 0.32.15.
- Existing Pi model configuration: 32,768-token context, 4,096 maximum response.
- Existing model parameters: temperature 1, top_k 64, top_p 0.95.
- Pi requested medium thinking, but this provider disables `reasoning_effort`;
  it is not equivalent to the hosted model's reasoning budget.
- 24 GiB unified memory, 12 logical CPUs. Analysis requests used loopback Ollama.
- No model downloads, global configuration edits or package source changes.
- Same locked scientific environment and exact input bytes as the hosted pilot.

A real read-tool smoke test passed first. The current hosted assistant prepared
and evaluated the experiment but did not repair the local agents' code. This
is not a claim that the entire supervisory workflow ran locally.

## 1. Full-study replication: zero completed studies

The [protocol](PROTOCOL.md) was written before the study agents started. Same
hierarchical Gaussian fixture and numerical settings as the hosted pilot:
160 observations, eight clinics, a fixed associational estimand beta, float64,
four chains, 500 warmup and 1000 retained draws. Each initial session had 15 minutes.
Initial runs were sequential, in seeded order A → C → B, to avoid Ollama contention.

| Setup | Session duration | Tool calls | Highest reported input tokens | Outcome |
|---|---:|---:|---:|---|
| A: NumPyro + ArviZ | 10m 31s | 10 | 32,215 | Length termination; no code, fit or report |
| B: direct Bayesjax | 4m 47s | 9 | 32,065 | Length termination; no code, fit or report |
| C: Bayescycle CLI | 3m 41s | 8 | 30,980 | Length termination; no code, fit or report |

All three repeatedly inspected directories and raw numerical JSON. A and B read
the main data more than once (A also printed the complete one-line file through
grep); C read both main and recovery data. No scientific API was actually used.
No compaction-start event was observed in any trial. A `length` stop near the
configured context limit is evidence of a context/length bottleneck; it does not
by itself identify every internal Ollama/Pi cause.

Additional errors:

- A used bare `python3`, ignoring the supplied `uv`/shared-environment instruction,
  and got `ModuleNotFoundError: numpyro`. NumPyro was present in the specified
  scientific environment; this was not evidence that its installation was broken.
- C tried an incorrect relative package-doc path and got a filesystem error.
- B had no flagged tool error, but still produced nothing. Process exit code zero
  was not counted as scientific success.

There were no accepted initial outputs to revise or hand off, so those stages
were **not run**, not silently passed. The shorter failed-session durations do
not make B/C faster or better. These failures happened before library interaction,
so this part did not isolate the value of the scientific interfaces.

## 2. Exploratory compact authoring probe

After observing the broad-task failures, a [separate probe](AUTHORING-PROTOCOL.md)
was registered before either new probe session. This is an openly post-hoc
refinement, not a replacement for the failed primary experiment.

Each fresh session got the same model mathematics and compact data shapes, a
short assigned-library API card with a scalar example, and one task: write
`model.py`. No raw data or hosted solutions were available in these workspaces.
Only read/write/edit tools were enabled, and each session had four minutes.
There was no CLI authoring arm because B and C share the Bayeswire declaration API.

| Interface | Duration | Tool calls | Outcome |
|---|---:|---:|---|
| NumPyro | 3m 19s | 6 | Model file produced; source and numerical checks passed |
| Bayeswire | 4m 00s | 3 attempted | No model; response-length limit, then deadline |

NumPyro's actual file is [a-model/model.py](a-model/model.py). It implements the
specified priors and non-centered likelihood without evaluator edits.

**The evaluator, not Gemma, ran the numerical validation.** The separate
`validate_authoring.py` driver bound the fixed main data and ran the original
NUTS settings. All saved parameter arrays exactly match the hosted NumPyro
reference, not just the displayed beta summary:

- beta mean: **0.5880398931**
- equal-tailed 95% interval: **[0.4729374928, 0.6984312517]**
- maximum all-parameter rank Rhat: **1.0031244**
- minimum bulk / tail ESS: **707.37 / 731.16**
- retained divergences: **0**

Bayeswire's probe read the brief and API card, then generated an attempted
`read(path=".")` tool call at a response-length limit. Pi refused to execute that
potentially truncated call. No model file existed at the four-minute deadline.
Its highest reported input count was only 2,596 tokens: this was **not** the same
large-input/context-pressure pattern as the full-study trials. It exhausted a
response budget and then the wall-time budget without delivering the file.
There was no model to compile or validate; the evaluator did not write one for it.

The compact probe changed task size, information presentation and available tools
at once. It shows that this supported local model could author the NumPyro model;
it cannot establish which change enabled success or an intrinsic API error rate.
Agent familiarity with NumPyro, stochastic decoding, the particular reference
cards, and the chosen budgets are important limitations.

## What this means for the project

1. **Simply replacing the hosted model with the installed local one did not make
   the existing workflow reliable.** All full tasks stopped before doing science.
2. **We still have no observed local-agent advantage for the custom interface.**
   NumPyro was the only successful compact authoring attempt. Do not retry or
   selectively change budgets until the preferred library wins.
3. **This does not invalidate the custom inference implementation.** The hosted
   comparison already exercised it successfully; the local failures did not even
   reach its sampler. Here the failures were agent execution and budget failures.
4. **The next useful design pressure is on the agent-facing task boundary:**
   compact schema/data summaries instead of raw arrays, small phases, executable
   checks, and honest stopping states. This is consistent with valuing explicit
   phases, not proof that the five-package/process architecture is necessary.
5. **Human control remains unmeasured.** Neither a prescribed-model task nor an
   evaluator-run fit establishes a successful human-led research experience.

No Tau, PydanticAI or MCP integration was built or tested, and no inference
algorithm, model configuration or library implementation was tuned after failures.

## Inspect the evidence

- [PROTOCOL.md](PROTOCOL.md): full-study replication and fixed budgets.
- [AUTHORING-PROTOCOL.md](AUTHORING-PROTOCOL.md): follow-up scope and confounds.
- [AUTHORING-BRIEF.md](AUTHORING-BRIEF.md): shared compact model specification.
- [NumPyro API card](a-model/API.md) / [Bayeswire API card](b-model/API.md).
- `evaluator/assessment.json`: session statuses, failed calls, length stops,
  missing outputs and the separately labeled authoring validation.
- `evaluator/local-environment.json`: installed model digest/configuration metadata.
- `evaluator/<arm>/<stage>/tool-calls.json`: actual calls, without thinking text.
- `evaluator/<arm>/<stage>/execution.json`: measured duration and process outcome.
- `evaluator/a-model/validation/`: evaluator-owned fit, original source/input
  hashes, retained draws, sampling statistics and recomputed diagnostics.
- Raw session/event streams are retained locally and gitignored; they are not
  bundled into the human-facing report.

Recompute the assessment without inference:

```bash
uv run --project experiments/workflow-value-pilot python \
  experiments/gemma-workflow-pilot/assess.py
```

The validation driver refuses existing output directories and requires an explicit
source-review flag. Do not overwrite evidence to rerun a failed attempt.
