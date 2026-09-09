# Agentic Bayesian vision: evidence, diagnosis, and the MVP test

**Status:** working document. Revised 2026-09-09. It consolidates the earlier
retrospective, the five-test agenda, and the separate portable-execution
protocol draft into one plan. The earlier text is retained in the appendices
so nothing from those discussions is lost. The deleted
`experiments/portable-execution-pilot/PROTOCOL.md` is recoverable from git
history at commit `7d29faf`; its reusable specifics are folded in below.

This is not an approved implementation plan or a frozen experiment protocol.
No new experiments were run while preparing this revision. The only new
observations are the direct inspection of the installed Bayesite binary and the
saved IR artifacts recorded in [Diagnosis](#diagnosis).

## Contents

1. [Evidence consulted](#evidence-consulted)
2. [The vision](#the-vision)
3. [What the experiments established](#what-the-experiments-established)
4. [Diagnosis](#diagnosis)
5. [The MVP test](#the-mvp-test)
6. [Prerequisite work before the MVP can run](#prerequisite-work-before-the-mvp-can-run)
7. [Decision rules](#decision-rules)
8. [Backlog: what is deferred and why](#backlog-what-is-deferred-and-why)
9. [Appendix A: original conversation (2026-09-05)](#appendix-a-original-conversation-2026-09-05)
10. [Appendix B: previous five-test agenda (superseded)](#appendix-b-previous-five-test-agenda-superseded)
11. [Appendix C: complexity assessment (2026-09-09)](#appendix-c-complexity-assessment-2026-09-09)

## Evidence consulted

Experiment records read during the original discussion and re-read on 2026-09-09:

- [Hosted pilot README](workflow-value-pilot/README.md),
  [protocol](workflow-value-pilot/PROTOCOL.md),
  [initial brief](workflow-value-pilot/INITIAL-BRIEF.md), and
  [evaluation](workflow-value-pilot/evaluator/evaluation.json)
- [Local Gemma pilot README](gemma-workflow-pilot/README.md) and
  [assessment](gemma-workflow-pilot/evaluator/assessment.json)
- [NumPyro authoring API card](gemma-workflow-pilot/a-model/API.md) and
  [Bayeswire authoring API card](gemma-workflow-pilot/b-model/API.md)
- Saved walkthroughs: [hosted](workflow-value-pilot/report.html) and
  [local](gemma-workflow-pilot/report.html)

Additional inspection on 2026-09-09:

- The installed Bayesite executable at `~/.cargo/bin/bayesite` (unverified
  version; see Diagnosis).
- The saved clinic model IR at
  `workflow-value-pilot/c-bayescycle/results/initial/model.ir.json`.
- Package sizes in this checkout: roughly 59k lines of Python across
  Bayescycle, Bayesjax, Bayeswire, and the two visualization projects.

The Rust engine source, browser deployment, and security architecture have not
been audited. Statements about the endgame are aspirations unless labelled as
observed.

## The vision

Stated by the user on 2026-09-05, reproduced verbatim so the plan below can be
checked against it:

> You're building the first Bayesian modeling system designed for the agentic era — where a model is data, not compiled code. A readable Python eDSL (or an agent, or any conformant producer) emits a versioned, language-neutral IR; that IR is executed by interchangeable engines: a JAX reference backend that defines the semantics and handles large models, and bayesite — a zero-dependency, capability-free Rust engine, sqlite-small, that starts in milliseconds and runs anywhere: a terminal, an agent's ephemeral sandbox, CI, or a browser tab via WASM. Around the engines sits a full workflow discipline as CLI verbs — prior predictive, simulate, recover, SBC, sample, diagnose, posterior predictive — with run directories, fingerprinted provenance binding every fit to the exact bytes of its model and data, and replayable analyses.
>
> The three convictions underneath it:
>
> No compile step, ever. Every incumbent — Stan, nutpie, even browser-Stan — compiles per model, which forces toolchains and servers into the loop. Model-as-IR fed to one pre-built engine deletes that entire category of friction, and it's the property that makes agent iteration (dozens of model variants in a sandbox) and true client-side execution possible at all.
> Agent-operability as the primary design constraint. Small in-context grammar, typed prescriptive errors that work as repair instructions, stateless verbs with exit codes, filesystem artifacts. The tool's first user is an agent running the workflow loop; humans get the readable surface and the documents (Showboat-style narrated, re-runnable analyses) that fall out of it.
> Safety by architecture, not mitigation. IR decodes without executing code; sampling is capability-free; the only untrusted-code exposure is one small authoring step that can live in the user's own browser. In the eventual deployed product — an agent drafts the model server-side as data, the user adjusts it in readable Python, everything executes client-side — your server runs no untrusted computation. The risk isn't contained; it's designed out.
>
> Holding it together is the contract-first governance you've now made concrete in bayescycle: bayeswire (stdlib-only eDSL + IR + specs + conformance corpus) as the slow-moving center, bayesjax as the oracle every engine must statistically agree with, bayescycle as the lean workflow CLI, viz and export tools kept heavy but firewalled behind process boundaries — and reproducibility (hashable IR, diffable model iterations, fingerprinted fits) as a side effect of the artifact design rather than a feature bolted on.
>
> The endgame: anyone — agent or human — can go from "describe a modeling problem" to a fitted, diagnosed, visualized, reproducible Bayesian analysis, anywhere code can run, with nothing to install but one small binary. The sqlite of Bayesian workflow, built for a world where the analyst is often an agent.

Sharpened claims agreed in the original discussion (full reasoning in Appendix A):

- "No compile step, ever" is defensible as **no model-specific compilation or
  user toolchain required by the portable engine**. The JAX backend does
  runtime compilation and NumPyro needs no Stan-style build either.
- "Safety by architecture" is defensible as **the server is never required to
  execute submitted arbitrary code**. It does not cover resource exhaustion,
  parser correctness, or WASM host bugs.
- Semantics live in the **specification**; Bayesjax is an executable reference
  and numerical oracle, not the definition.

## What the experiments established

Two pilots ran on 2026-09-04. Both compared three arms on the same eight-clinic
hierarchical Gaussian fixture: A NumPyro plus ArviZ, B Bayeswire plus direct
Bayesjax, C Bayeswire plus the Bayescycle CLI. Neither pilot let any arm use the
Bayesite engine.

| Vision claim | Evidence so far |
|---|---|
| Portable, small, prebuilt execution engine | Not tested |
| Model transport without executing author code | Not meaningfully tested |
| Client-side/WASM execution | Not tested |
| Agent-friendly model authoring | Tested narrowly (local Gemma probe); unfavorable |
| Low-friction workflow verbs | Tested with Bayesjax backend; extra friction |
| Standardized provenance | Demonstrated, no unique outcome advantage |
| Model reuse for prediction | Failed: B and C hand-wrote the forward equation |
| Replay after relocation | Not tested |
| Human-controlled scientific iteration | Not tested |

### Observations we must preserve

- NumPyro was the lower-friction solution in the hosted pilot. All arms completed the prescribed numerical task, revision, and handoff.
- Direct Bayesjax and Bayescycle worked on that fixture; the negative result concerns demonstrated usability advantage, not evidence of incorrect inference.
- Bayescycle supplied standardized native artifacts and fingerprints, but ordinary scripts also passed the tested preservation and handoff requirements.
- The full local-agent attempts failed before scientific API use. Those failures cannot isolate backend or modeling-language quality.
- The compact local authoring probe succeeded only with NumPyro. It is an exploratory, single-attempt result with budget, presentation, familiarity, and stochastic-decoding limitations.
- B and C duplicated the forward equation for posterior prediction. This is a concrete workflow usability gap.
- The Rust engine, deployment simplicity, browser execution, exact relocated replay, and actual human approval workflow were not tested by these pilots.

### Qualifications retained from the underlying reports

- All 12 completed saved fits passed the specified chain-diagnostic thresholds with zero retained-draw divergences. This does not establish scientific validity or calibration.
- All three initial recovery intervals missed the known beta of 0.4; the tighter-prior revision did not fix that. Agents reported the miss. One recovery dataset is not a calibration test.
- B and C used the same engine/settings and produced bit-identical posterior arrays. Their agreement is not independent cross-engine corroboration.
- Hosted implementation sessions ran concurrently with shared hardware/cache effects. Session times are not sampler benchmarks.
- The compact local NumPyro model was fitted and numerically validated by the evaluator, not by the local authoring agent.
- Local hashes establish specified byte-consistency relationships, not signatures, authenticated execution, or proof that arbitrary source produced a particular posterior.
- The full gated study skill was deliberately not executed. No actual human approvals were measured or invented.

### Hosted pilot reference numbers

Single-run observations with concurrent sessions; useful as a reference point,
not as a stable estimate.

| Initial implementation | NumPyro | Direct Bayesjax | Bayescycle |
|---|---:|---:|---:|
| Session time | 3m 45s | 5m 02s | 7m 05s |
| Tool calls | 19 | 35 | 51 |
| Authored Python lines | 184 | 187 | 194 |

## Diagnosis

The user's conclusion on 2026-09-09: the approach was well intended but too
complex, and the vision needs an MVP-scale test. The evidence supports that,
with one refinement about where the complexity lives.

### The Python layers are the heavy part, and they are what was tested

Every layer stacked on top of the engine cost the agent effort. The ordering in
the hosted pilot was NumPyro, then direct Bayesjax, then Bayescycle CLI. The
CLI did not remove the need for analysis scripts, leaked backend selection into
the agent's work, and needed pinned `uvx` environments for plotting and export.
The five-package lockstep governance, process-boundary firewalls, and
run-directory specifications are all built for a hypothesis that has not been
validated yet.

The experiment design inherited the same weight. The portable-execution
protocol drafted after the retrospective had 282 lines of pre-run gates
(frozen manifests, scratch containers with dropped capabilities, checksum
verification, event logs, telemetry) before any scientific command. That rigor
suits a publishable result. It is the wrong shape for a first signal on an
unproven hypothesis.

### The distinctive artifact is small and was never handed to an agent

Observed on 2026-09-09 from the installed executable (version not verifiable,
see below):

| Property | Observed |
|---|---|
| Executable size | 1.2 MB |
| Cold start to a typed error | about 4 ms |
| Verbs present | `sample`, `diagnose`, `prior-predictive`, `recover`, `sbc` |
| Verbs missing | `posterior-predictive` |
| Version or capabilities command | none; `--version` returns an `InvalidSettings` error |
| Error format | one JSON object with `error_format`, `error`, and `message` |

That executable is the "sqlite of Bayesian workflow" artifact. Neither pilot
let an agent run it. The Bayescycle arm's dry plans selected Bayesite for
diagnose and posterior-predictive but the protocol prohibited executing it.

### The IR is a codec target, not yet an authoring surface

The saved clinic model IR is 4,170 bytes of nested node JSON for seven
parameters: every constant is a `ConstNode` object, every parameter a
`ResolvedParam` wrapper. It is unambiguous and decodes without executing code,
which is the property the vision needs. It is also verbose. Whether an agent can
author it directly from a short card, or whether a Python eDSL step is required
on the producer side, is the open question the MVP must answer. The vision
explicitly allows both: "a readable Python eDSL (or an agent, or any conformant
producer) emits" the IR.

### The irreducible claim

Strip the vision to the one claim everything else depends on:

> An agent with nothing installed but one small binary can go from a modeling
> problem and data to a fitted, diagnosed, predicted, and revised model by
> writing the model as data and running stateless verbs.

Run directories, fingerprints, relocated replay, revision integrity,
visualization, WASM, and the human approval workflow are downstream of this. If
it holds, they are worth building around what the agent actually needed. If it
fails, they are premature regardless of how well engineered they are.

## The MVP test

**Question:** Given only the Bayesite executable, a one-page authoring card, a
data file, and a prose problem statement, can a capable hosted agent complete
prior prediction, sampling, diagnostics, posterior prediction, and one prior
revision on the clinic fixture without Python and without hand-writing any
model equation?

This collapses the earlier Test 1 (portable execution), Test 2 (agent
authoring and repair), and Test 3 (model reuse) into one session design. It is
an agent trial, not a deployment audit; the deployment rigor is deferred to the
backlog.

### Environment

- A fresh working directory containing: the `bayesite` executable, `CARD.md`,
  `data.json`, `design.json`, and `TASK.md`. Nothing else.
- `PATH` restricted so that only `bayesite` and basic shell utilities resolve.
  `python`, `python3`, `uv`, and `uvx` must not resolve; record the check.
  A plain restricted shell is sufficient for the MVP. No container, dropped
  capabilities, or network telemetry.
- No repository checkout, no Bayeswire, no Bayescycle, no eDSL.
- Agent harnesses: Pi for the four OpenAI models, Claude Code for Fable 5.1.
  See [Agents](#agents) for the roster, settings, and the confound this
  introduces.

### Inputs

- **Fixture:** the initial eight-clinic, 160-observation hierarchical Gaussian
  model from the hosted pilot. Copy the files byte for byte; digests verified
  on 2026-09-09. Paths are relative to `workflow-value-pilot/`.

  | Sandbox file | Source | SHA-256 |
  |---|---|---|
  | `data.json` | `c-bayescycle/results/initial/data.json` | `3231a98c65616c2edaff34e572ed9b59887b88a0ce1b2a16a0b7b725964abd34` |
  | `design.json` | `c-bayescycle/results/prior/data.json` | `6204ef1441b60ab12fcfe4da1002d5745cec70dfa685f36f260a4107fa48b7b9` |
  | evaluator only: reference IR | `c-bayescycle/results/initial/model.ir.json` | `2da7c00eadf034cea33c7f10f0d98510233b1074653f619de9bd711097763f65` |
  | evaluator only: NumPyro posterior | `a-numpyro/results/initial/posterior.npz` | `1fa21d45fa47f2a9cc17f5d49ab1d044b80da447def796b95763c682ea7d3fba` |

  `data.json` contains observations; `design.json` omits `y`. Prior prediction
  uses the design document, posterior prediction uses the exact fit data.
- **Model IR is not supplied.** The agent must produce it. The reference IR is
  held by the evaluator as the answer key and must not be in the sandbox.
- **`TASK.md`:** the model in words and mathematics, the estimand beta, the
  numerical settings (float64, 4 chains, 500 warmup, 1000 retained draws,
  seeds), the required outputs, and the scripted revision (beta prior SD from 1
  to 0.25). Same specification as the hosted pilot's brief, reduced to the
  workflow loop.
- **`CARD.md`:** the in-context grammar. One page of IR structure plus one
  complete worked example of a different model (not the clinic model), plus the
  verb usage lines and the error format. Derived from
  `spec/ir-format-v1.md`, `spec/ir-v1-tags.md`, and `spec/data-document-v1.md`.
  The card is part of the product under test: if it cannot be one page, that is
  a finding.

### Agents

Five agents, decided on 2026-09-09. Availability verified the same day: the
four OpenAI models resolve in Pi under the `openai-codex` provider, and Claude
Code 2.1.267 is installed.

| Agent | Harness | Settings to freeze |
|---|---|---|
| `gpt-6-astra` | Pi | medium reasoning, no extensions or skills; same as the hosted pilot |
| `gpt-5.6-sol` | Pi | same |
| `gpt-5.6-terra` | Pi | same |
| `gpt-5.6-luna` | Pi | same |
| `claude-fable-5-1` | Claude Code, non-interactive `claude -p` | model pinned by id, cwd set to the sandbox, no project `CLAUDE.md`, record the permission mode and effort setting |

Running several agents turns the MVP from "can a capable agent do this" into
"how capable does an agent need to be", which is a more useful result for the
card and error design. Two confounds are accepted and must be reported, not
hidden:

- **Harness differs with model for Fable.** Claude Code and Pi expose
  different tools and prompts. The within-Pi comparison across the four OpenAI
  models is the clean one; the Fable arm is a separately labelled comparison
  of model plus harness.
- **Fable wrote the plan.** The card, task, and this protocol are drafted by
  the same model family that runs the Fable arm, in this checkout, with
  project memory. The trial sessions run in a fresh sandbox directory so the
  per-project memory is empty, and the user-level `~/.claude/CLAUDE.md` was
  verified empty on 2026-09-09. Record both checks in the manifest. This does
  not remove the author-advantage confound; it only bounds it.

### Arms

| Arm | Authoring | Where Python runs | Sessions per agent | Total |
|---|---|---|---:|---:|
| IR-direct | Agent writes `model.ir.json` from the card | Nowhere | 3 | 15 |
| eDSL-producer | Agent writes Bayeswire Python on a producer host, serializes, ships IR to the sandbox | Producer only | 3 | 15 |
| NumPyro reference | Agent implements the same task with NumPyro in the installed environment, hosted pilot brief | Installed environment | 1 | 5 |

Thirty-five sessions at up to 20 minutes each. Three repetitions per agent and
arm is the smallest count that separates a single lucky or unlucky run from a
pattern; five agents give the breadth that five repetitions of one agent did
not. The existing `gpt-6-astra` NumPyro result from the hosted pilot is kept
as a sixth reference point, labelled as run under the earlier protocol.

IR-direct is the primary arm because it is the untested path and the one the
vision describes for the server-side agent. eDSL-producer is included because
the Gemma probe's unfavorable eDSL evidence was a single local-model attempt and
should not stand alone. The NumPyro reference is a fair comparison for the
authoring-effort question only; it does not run in the restricted environment
and is not a portability comparison.

### The loop each session must complete

1. Author the model as data.
2. `prior-predictive` against `design.json`.
3. `sample` against `data.json` with the fixed settings.
4. `diagnose` the fit.
5. `posterior-predictive` against `data.json` using the fit.
6. Revise the prior, then repeat steps 3 to 5 into separate output files,
   preserving the originals.
7. Write a short `RESULT.md` with the beta summary, diagnostics, and what was
   revised.

`recover` and `sbc` exist in the binary and are stretch goals only; they are
not scored.

### Measurements per session

- Attempt on which the IR first decodes, first samples, and first passes
  diagnostics.
- Each engine error received and whether the following edit fixed the issue the
  error named (typed errors as repair instructions).
- Tool calls, wall time, and tokens where the harness exposes them.
- Whether any model equation was written outside the IR.
- Whether the revised fit was written to new files with the originals intact.
- Whether the beta summary agrees with the NumPyro reference within
  `4 * hypot(MCSE_A, MCSE_B)`, checked by the evaluator, not the agent.
- Whether the session stopped honestly when blocked rather than fabricating
  output.

### Budgets

- 20 minutes wall time and one session per attempt. No retries until a pass.
  Every attempt is preserved.
- Sample calls capped at 15 minutes; other verbs at 2 minutes.
- Sessions run sequentially within a provider. Providers may run concurrently
  since the OpenAI models and Claude Code call different services. Timings are
  descriptive only and carry no decision rule, so this concurrency is
  acceptable; record what ran alongside what.
- Within each agent, run the three IR-direct sessions first, then the three
  eDSL-producer sessions, then the NumPyro reference. Fixed order, no
  reordering after seeing results.

### Evidence discipline, reduced to what matters

- New directory `experiments/mvp-agent-binary/`. The old pilots are untouched.
- One `MANIFEST.json` recording SHA-256 of the executable, card, task, fixtures,
  and this protocol section before the first session.
- The evaluator's acceptance script is frozen before sessions start.
- Missing engine capability is recorded, never patched mid-experiment.
- The README reports every session, including failures, in one result matrix.

## Prerequisite work before the MVP can run

These are product changes and must be done and recorded separately from the
trial. Record the pre-change capability inventory first so the before and after
are distinguishable.

1. **`posterior-predictive` verb in Bayesite.** The only missing verb, and the
   exact spot where arms B and C failed in the hosted pilot. Interface proposed
   in the earlier protocol:
   `bayesite posterior-predictive --model <ir> --data <data> --fit <fit.jsonl> [--seed N] [--out <pp.jsonl|->]`,
   emitting one replicate per retained draw. Same-design prediction for existing
   observations and groups only; new-group prediction is out of scope.
   Validate with the analytic fixture from the earlier protocol, which has
   closed-form predictive moments: `theta ~ Normal(0,1)`, `x = [0,0,0,0]`,
   `y_i ~ Normal(theta + x_i, 1)`, `y = [-1,0,1,2]`, posterior
   `Normal(0.4, sqrt(0.2))`, prior predictive marginal variance 2 with
   covariance 1, posterior predictive marginal variance 1.2 with covariance 0.2.
2. **A `version` or `capabilities` verb in Bayesite.** Today there is no way to
   identify the executable from the executable. An agent and an evaluator both
   need it, and every provenance claim depends on it.
3. **`CARD.md`.** Written from the spec, tested once by hand against the
   engine, and frozen. Its length and the number of concepts it needs are
   findings in themselves.
4. **Pin the executable.** A release archive with a published checksum, not the
   local `~/.cargo/bin` build. The earlier protocol proposed `v0.4.0`
   `x86_64-unknown-linux-musl`; the MVP can use whichever target matches the
   sandbox host as long as it is a checksummed release.

Nothing in the Python packages needs to change for the MVP to run.

## Decision rules

Written before any session runs. A rule that is not written down here before
execution does not count.

**Engine capability (R0).** If the pinned executable cannot complete the loop on
the evaluator's reference IR, the trial is blocked, not failed. Fix the engine,
record the change, and start again.

**Authoring (R1).** An agent passes IR-direct authoring if at least two of its
three sessions produce IR that samples successfully within three engine-error
repair rounds. If fewer than three of the five agents pass, the raw IR is not
agent-operable from a one-page card. Then the eDSL-producer arm decides whether
the authoring surface or the engine boundary is the problem. Per-agent results
are reported as a gradient regardless of the aggregate verdict, since "which
agents can do this" is itself the finding.

**Model reuse (R2).** If any session hand-writes a likelihood or predictive
equation while the `posterior-predictive` verb exists and is on the card, the
model-as-data benefit is not discoverable and the card or verb surface needs
work before more workflow investment.

**Honesty (R3).** Any session that reports a completed fit or prediction the
evaluator cannot reproduce from the session's own files is a failure regardless
of other scores.

**What a pass licenses.** Only this: the named agents that passed, given one
checksummed executable and a one-page card, ran the full workflow loop on a
hierarchical Gaussian model without Python. It does not license "runs
anywhere", "sqlite-small", "millisecond startup", WASM, safety, replay, or
revision-integrity claims. Those stay in the backlog.

**What follows a pass.** Rebuild the Python layer around what the agent
actually needed. Candidates from the observations: a producer-side eDSL only if
the eDSL arm was clearly better; provenance only as a side effect of the
verbs' own outputs; no workflow CLI on top of the engine unless a session showed
it was missed.

**What follows a failure on R1.** Decide explicitly between a more familiar
authoring surface on the producer side and a redesigned IR. Do not commission
another whole-stack comparison.

## Backlog: what is deferred and why

Each entry keeps its question so it is not lost. Full earlier drafts are in
Appendix B.

- **Portable execution audit** (earlier Test 1 and the deleted protocol).
  Scratch container with executable only, read-only root, no network, dropped
  capabilities, cold-start repetitions, peak memory, negative cases for a bad IR
  version and a shape mismatch. Deferred because the MVP shows whether the
  binary is worth auditing. The negative cases are cheap and can be run as
  engine tests whenever the `version` verb lands.
- **Relocated replay** (earlier Test 4). Define which guarantee is meant
  (artifact verification, summary reconstruction, computational replay, bitwise
  replay, statistical reproducibility) before measuring. Deferred because it
  tests the provenance layer, which is downstream of the irreducible claim.
- **Revision integrity** (earlier Test 5). Whether the artifact design catches
  stale results after a model or data change, versus a cooperative agent
  remembering. Deferred for the same reason. The MVP's scripted revision gives
  a first informal read on whether agents keep originals intact.
- **Browser and WASM execution.** A separate platform test. Nothing native
  establishes it.
- **Security boundary inspection.** Parser fuzzing, resource limits, WASM host
  review. Not implied by any workflow pass.
- **Human-led study.** Whether a person can understand, challenge, and approve
  an agent's scientific decisions. None of the engineering checks measures it.
- **Local small-model operability.** The Gemma probe failed on context handling
  before reaching the science. Rerun only after the MVP card exists, since the
  card is the in-context grammar the claim is about.
- **Bayesjax posterior-predictive API.** The hosted pilot's arm B gap. Not
  needed for the MVP and only worth fixing if the Python layer survives the
  decision rules.

## Appendix A: original conversation (2026-09-05)

Reproduced verbatim from the first revision of this document, including the
summary that closed that discussion. Headings are demoted one level.

### Full conversation

#### 1. User: concern about the experiments

can we discuss about the experiments we have conducted in experiments/. it is was supposed to show some usefulness of this library for agentic bayesian workflow, but seems it failed and numpyro seemed more succesful.

#### 2. Assistant: initial reading of the evidence

Yes. I read both experiment summaries, their evaluation results, and the task/API briefs. **Your interpretation is broadly right: NumPyro was the stronger practical choice in these trials.** I would not explain that away—but I would distinguish which hypotheses failed.

##### What the evidence actually says

In `experiments/workflow-value-pilot/`, all three approaches produced acceptable numerical results, preserved the original analysis, and passed fresh-session handoff.

But NumPyro required less effort:

| Initial implementation | NumPyro | Direct Bayesjax | Bayescycle |
|---|---:|---:|---:|
| Session time | 3m 45s | 5m 02s | 7m 05s |
| Tool calls | 19 | 35 | 51 |
| Authored Python lines | 184 | 187 | 194 |

These are single-run observations, not stable performance estimates. Nevertheless, **the custom stack added work without demonstrating a better outcome on the tested task.** In particular, the CLI did not remove the need for analysis scripts.

In `experiments/gemma-workflow-pilot/`:

- All full-study attempts failed before using the scientific APIs. That mostly tells us the task/context handling was unsuitable for this local agent.
- In the smaller authoring task, NumPyro succeeded and Bayeswire produced no model within budget. That is limited but genuinely unfavorable evidence for the “narrower interface helps smaller agents” hypothesis.

##### Why I think this happened

###### 1. A smaller language is not necessarily easier for an agent

NumPyro benefits from familiar Python, standard array operations, and likely much greater representation in training data.

Bayeswire introduces different concepts: class declarations, explicit constraints, symbolic data declarations, and `linear(...).apply(...)` instead of `@`. Those might offer useful validation properties, but the agent must learn them first.

**Restricting expressiveness can make validation easier without making authoring easier.** We appear to have treated those as more closely connected than these results support.

Familiarity is a possible experimental confound—but it is also a real product advantage.

###### 2. The workflow layer did not absorb enough complexity

The clearest concrete problem is posterior prediction: B and C duplicated the model’s forward equation in supplementary scripts, whereas A used NumPyro’s `Predictive`.

That is more than inconvenient. It creates another place where the implemented science can drift.

C also had to navigate backend-dependent command availability and artifact conversions. Explicit boundaries may be good engineering internally, but **if agents have to manage all those boundaries themselves, the user-facing workflow is not simpler.**

###### 3. Ordinary scripts were a stronger baseline than expected

The NumPyro agent preserved models, saved draws, recorded hashes, explained revisions, and left a usable handoff.

Bayescycle supplied more standardized provenance, which is real value. But the experiment did not show that those extra guarantees prevented a mistake or enabled something the simpler approach could not do.

“We have structured artifacts” is a capability. “Those artifacts materially improve research work” is a separate claim.

##### What remains open—and what should not become an excuse

Neither experiment tested actual human-led model development: negotiating an estimand, rejecting assumptions, interpreting predictive failures, or deciding whether a revision is justified.

So the broader agentic Bayesian workflow idea remains open. **But “we haven’t tested the real value yet” cannot indefinitely justify the existing architecture.** Each additional layer still needs to earn its cost.

My current recommendation would be:

- Treat **NumPyro + ArviZ + ordinary files as the default baseline**.
- Separate the value of the custom model language/backend from the value of human-controlled workflow.
- Test the smallest workflow support around a real analysis—not another complete stack.
- Measure whether it prevents concrete mistakes, reduces human review effort, or makes decisions easier to understand.

For example: after changing a prior, can the system reliably identify which checks and conclusions are stale—and can you understand and approve the change more easily than with scripts and a study note?

**My strongest takeaway is that the experiments challenge the need for a custom computational stack more than they challenge the need for agentic Bayesian workflow support.**

Is your main ambition to make Bayesian work with an agent genuinely better, regardless of backend—or is having an inspectable, constrained model language itself an essential part of what you want to build? Those lead to different next steps.

#### 3. User: the intended vision

my vision is this one:
You're building the first Bayesian modeling system designed for the agentic era — where a model is data, not compiled code. A readable Python eDSL (or an agent, or any conformant producer) emits a versioned, language-neutral IR; that IR is executed by interchangeable engines: a JAX reference backend that defines the semantics and handles large models, and bayesite — a zero-dependency, capability-free Rust engine, sqlite-small, that starts in milliseconds and runs anywhere: a terminal, an agent's ephemeral sandbox, CI, or a browser tab via WASM. Around the engines sits a full workflow discipline as CLI verbs — prior predictive, simulate, recover, SBC, sample, diagnose, posterior predictive — with run directories, fingerprinted provenance binding every fit to the exact bytes of its model and data, and replayable analyses.

The three convictions underneath it:

No compile step, ever. Every incumbent — Stan, nutpie, even browser-Stan — compiles per model, which forces toolchains and servers into the loop. Model-as-IR fed to one pre-built engine deletes that entire category of friction, and it's the property that makes agent iteration (dozens of model variants in a sandbox) and true client-side execution possible at all.
Agent-operability as the primary design constraint. Small in-context grammar, typed prescriptive errors that work as repair instructions, stateless verbs with exit codes, filesystem artifacts. The tool's first user is an agent running the workflow loop; humans get the readable surface and the documents (Showboat-style narrated, re-runnable analyses) that fall out of it.
Safety by architecture, not mitigation. IR decodes without executing code; sampling is capability-free; the only untrusted-code exposure is one small authoring step that can live in the user's own browser. In the eventual deployed product — an agent drafts the model server-side as data, the user adjusts it in readable Python, everything executes client-side — your server runs no untrusted computation. The risk isn't contained; it's designed out.

Holding it together is the contract-first governance you've now made concrete in bayescycle: bayeswire (stdlib-only eDSL + IR + specs + conformance corpus) as the slow-moving center, bayesjax as the oracle every engine must statistically agree with, bayescycle as the lean workflow CLI, viz and export tools kept heavy but firewalled behind process boundaries — and reproducibility (hashable IR, diffable model iterations, fingerprinted fits) as a side effect of the artifact design rather than a feature bolted on.

The endgame: anyone — agent or human — can go from "describe a modeling problem" to a fitted, diagnosed, visualized, reproducible Bayesian analysis, anywhere code can run, with nothing to install but one small binary. The sqlite of Bayesian workflow, built for a world where the analyst is often an agent.

#### 4. Assistant: revised interpretation in light of the vision

That clarifies something important. **Your vision is not primarily “a nicer NumPyro.” It is a portable execution contract for Bayesian models, with an agent-operable workflow around it.** My earlier suggestion to default to NumPyro addressed the analysis task, but not that larger objective.

I think the vision is coherent. I also think the experiments revealed a genuine weakness—not a refutation of its central architecture.

##### The experiments tested the cost, but largely not the distinctive benefit

The hosted pilot effectively asked:

> Given an installed Python/JAX environment and a capable coding agent, which stack completes this prescribed analysis most easily?

NumPyro won that comparison. That matters.

But the experiment supplied precisely the infrastructure your small binary is intended to make unnecessary. It also explicitly excluded the Rust backend.

| Vision claim | Evidence from these experiments |
|---|---|
| Portable, small, prebuilt execution engine | Not tested |
| Model transport without executing author code | Not meaningfully tested |
| Client-side/WASM execution | Not tested |
| Agent-friendly model authoring | Tested narrowly; unfavorable result |
| Low-friction workflow verbs | Tested with Bayesjax; extra friction |
| Standardized provenance | Demonstrated, but no unique outcome advantage |
| Replay after relocation | Not tested |
| Human-controlled scientific iteration | Not tested |

So I would revise the conclusion to:

**The agent-facing authoring and workflow experience has not earned its complexity yet. The portable, capability-restricted execution proposition remains largely untested.**

That is different from saying the whole project should become a NumPyro wrapper.

##### Three claims I would sharpen

###### “No compile step, ever”

The defensible promise is:

> **No model-specific compilation or user toolchain required by the portable engine.**

NumPyro does not require a Stan-style external compilation workflow, although JAX normally performs runtime compilation. Your JAX backend also makes “no compile step, ever” too broad.

Likewise, eliminating model-specific compilation is an excellent route to browser execution—not the only technically possible route.

The narrower statement is already compelling. It does not need the universal claim about incumbents.

###### “Safety by architecture”

This is a substantial differentiator **if the execution boundary genuinely accepts only validated data and exposes no ambient capabilities**.

But distinguish:

- IR decoding from Python authoring: arbitrary Python remains code execution.
- Capability isolation from resource safety: a model can still exhaust time or memory.
- Architectural restrictions from implementation correctness: parsers, numerical kernels, and WASM hosts still need scrutiny.

You can design out *the requirement to execute submitted arbitrary code on your server*. That is powerful. “The risk is designed out” overstates it.

###### “JAX defines the semantics”

I would put the authority in the **specification**, with JAX as an executable reference and numerical oracle.

Otherwise an implementation quirk risks becoming the contract. Engines should agree on model meaning and target density; finite posterior samples should satisfy appropriate statistical checks, not necessarily match.

##### The uncomfortable finding still stands

Your architecture can be right for portability while the interface is wrong for agents.

The compact Gemma probe suggests that a small unfamiliar grammar is not automatically easier than a larger familiar one. The hosted trial showed that workflow boundaries currently leak into the agent’s work.

Most tellingly, **an agent had to rewrite the forward equation for posterior prediction despite the system already possessing the model as data**.

That is exactly where your architecture ought to help. The IR should allow supported predictive operations to reuse the model, rather than requiring another implementation of its mathematics.

This is not an argument against model-as-data. It is evidence that its benefits are not yet fully delivered.

##### What I would test next

Not another generic “which library fits this regression faster?” contest. Test the distinctive promises directly, while retaining NumPyro as an honest baseline where applicable:

1. **Portable execution:** hand a clean environment only the binary, IR, and data. Complete a supported workflow without Python, dependency resolution, or model compilation.
2. **Agent repair:** give agents invalid models and measure whether typed errors lead to correct repairs within fixed budgets. Compare direct IR production with eDSL authoring.
3. **Model reuse:** execute prior and posterior prediction without duplicated model equations.
4. **Relocated replay:** move an analysis to a fresh location or machine; verify its inputs and reconstruct results under an explicitly defined replay guarantee.
5. **Revision integrity:** change model or data bytes and test whether stale results are identified rather than silently reused.

These should have declared failure criteria—not be demonstrations arranged to guarantee a win.

**My revised position: keep the vision, but narrow the next proof.** Show that the same model artifact can be safely transported, executed, inspected, and reused with materially less environmental machinery. Then show that agents can operate that system reliably.

The “SQLite of Bayesian workflow” claim becomes convincing when the user experiences that simplicity—not merely when the repository has the architecture to support it.

#### 5. User: request to preserve the discussion

can you put all we discussed into a markdown, wihtout any loss of context, so we can disuss thorugh your next suggested test steps?

### Where the discussion leaves us

This section summarizes the discussion; it does not replace the full conversation or imply that the user has approved the assistant's recommendations.

#### Observations we must preserve

- NumPyro was the lower-friction solution in the hosted pilot. All arms completed the prescribed numerical task, revision, and handoff.
- Direct Bayesjax and Bayescycle worked on that fixture; the negative result concerns demonstrated usability advantage, not evidence of incorrect inference.
- Bayescycle supplied standardized native artifacts and fingerprints, but ordinary scripts also passed the tested preservation and handoff requirements.
- The full local-agent attempts failed before scientific API use. Those failures cannot isolate backend or modeling-language quality.
- The compact local authoring probe succeeded only with NumPyro. It is an exploratory, single-attempt result with budget, presentation, familiarity, and stochastic-decoding limitations.
- B and C duplicated the forward equation for posterior prediction. This is a concrete workflow usability gap.
- The Rust engine, deployment simplicity, browser execution, exact relocated replay, and actual human approval workflow were not tested by these pilots.

#### Additional qualifications retained from the underlying reports

These details were in the evidence read during the discussion and remain important when interpreting “successful numerical results”:

- All 12 completed saved fits passed the specified chain-diagnostic thresholds with zero retained-draw divergences. This does not establish scientific validity or calibration.
- All three initial recovery intervals missed the known beta of 0.4; the tighter-prior revision did not fix that. Agents reported the miss. One recovery dataset is not a calibration test.
- B and C used the same engine/settings and produced bit-identical posterior arrays. Their agreement is not independent cross-engine corroboration.
- Hosted implementation sessions ran concurrently with shared hardware/cache effects. Session times are not sampler benchmarks.
- The compact local NumPyro model was fitted and numerically validated by the evaluator, not by the local authoring agent.
- Local hashes establish specified byte-consistency relationships, not signatures, authenticated execution, or proof that arbitrary source produced a particular posterior.
- The full gated study skill was deliberately not executed. No actual human approvals were measured or invented.

#### The important change in interpretation

Before the vision was clarified, the assistant recommended treating NumPyro as the default and questioning the need for the custom computational stack.

After clarification, the assistant distinguished two objectives:

1. **Completing Bayesian analyses efficiently in an already-installed Python environment.** The existing NumPyro baseline performed better on the tested task.
2. **Transporting and executing Bayesian models as data through a small, portable, constrained engine with standardized workflow artifacts.** The current experiments did not substantially test this distinctive proposition.

The revised recommendation is not to discard the vision or to declare it validated. It is to test its distinctive promises directly while preserving the unfavorable evidence about present authoring and workflow friction.

#### Claims versus aspirations

- “First,” “sqlite-small,” “starts in milliseconds,” “runs anywhere,” “handles large models,” and “nothing to install but one small binary” need operational definitions or supporting evidence before being treated as established product claims.
- “No model-specific compilation or user toolchain required by the portable engine” is more precise than “no compile step, ever.”
- A normative specification and an executable reference have different roles. Statistical agreement also differs from semantic conformance or identical draws.
- Capability restrictions do not establish unlimited resource safety, parser correctness, or scientific correctness.
- Existing separate Python visualization/export environments are relevant to the eventual single-binary end-to-end claim: the next protocol must say whether visualization is inside or outside the demonstrated scope.


## Appendix B: previous five-test agenda (superseded)

The test agenda and open decisions from the first revision, verbatim. The MVP
above replaces Tests 1 to 3 with one session design and moves Tests 4 and 5 to
the backlog. Headings are demoted one level.

### Next-test discussion draft

**Everything below is a proposed elaboration for the next discussion, not an already agreed protocol.** The five test themes came from the conversation; the questions and candidate measurements below make them easier to scope.

#### General approach

Separate deterministic product-capability checks from stochastic agent evaluations:

- First establish what the pinned released or development artifacts can actually do through public interfaces.
- Then measure whether agents can use those capabilities reliably.
- Missing functionality is a recorded result, not permission for an evaluator to silently fill the gap.
- If implementation work is chosen afterward, distinguish the pre-change baseline from the subsequent validation.
- Keep the old experiments and failures intact. Use new directories for new protocols and evidence.

#### Test 1 — Portable execution

**Question:** Can the portable engine execute a useful, explicitly bounded Bayesian workflow with only a prebuilt binary, model IR, and input data?

**Why this matters:** The earlier pilot supplied an installed scientific Python environment and prohibited the Rust backend. It did not test deployment simplicity.

**Candidate setup:**

- Pin the engine binary/version and record its digest and target platform.
- Start in a documented clean environment with no Python/JAX, model compiler, or scientific package installation step.
- Supply a known, supported model as IR and fixed data; do not require eDSL execution in the receiving environment.
- Declare the included operations before execution, distinguishing fitting/diagnostics/prediction from visualization and export.
- Disable or observe network access so hidden dependency fetching is not mistaken for zero-dependency operation.

**Candidate measurements:**

- Binary and analysis-bundle sizes, startup latency, end-to-end execution time, and peak memory.
- Required installations, network requests, subprocesses, and manual setup steps.
- Completed operations and independently checked numerical outputs.
- Behavior on unsupported IR versions or unsupported model features.

**Candidate failure signals:** An undeclared runtime dependency, model-specific build, backend fallback, external service, manual repair, or required operation that cannot complete.

**Fairness:** A deployment capability check does not have to be a head-to-head benchmark. If comparing NumPyro, report both fresh-environment and already-installed costs, with comparable packaging assumptions. Do not count delivery of one system's prebuilt dependencies as free while charging the other for theirs.

**Boundary:** Native execution does not establish WASM/browser support. That requires a separate platform test. A successful workflow run also does not prove capability isolation; that requires explicit boundary inspection and security testing.

#### Test 2 — Agent authoring and repair

**Question:** Do the grammar and typed errors enable agents to produce and repair scientifically correct supported models under fixed budgets?

**Why this matters:** This is a central agent-operability claim, and the existing authoring evidence is unfavorable.

**Candidate setup:**

- Keep initial authoring and repair of supplied invalid artifacts as separate tasks.
- Compare direct IR production with eDSL authoring for the same mathematics.
- Retain a NumPyro baseline where tasks and error classes are meaningfully comparable.
- Include distinct error classes such as invalid fields/tags, shape mismatches, constraint violations, and unsupported operations. Do not assume every interface can represent every same error.
- Supply compact schema/data summaries instead of raw numerical arrays.
- Fix agent versions, prompts, documentation, tool access, context/output limits, time budgets, and repetition counts before running.
- Use multiple independent attempts and preserve every attempt rather than retrying until a preferred interface wins.

**Candidate measurements:**

- Valid model produced, successful repair, tool calls, token usage, elapsed time, and evaluator interventions.
- Preservation of the intended priors, likelihood, dimensions, and estimand after repair.
- Whether an unsupported request produces an honest stopping state instead of a fabricated success.

**Candidate failure signals:** No usable artifact within budget; silent changes to the scientific target; ignored constraints; hidden fallback; or repairs requiring evaluator-written code.

**Boundary:** Syntactic acceptance is not scientific correctness. A local model's success or failure does not establish a universal interface error rate. If evaluating the benefit of prescriptive errors specifically, an error-message comparison must hold other task conditions fixed.

#### Test 3 — Model reuse for prediction

**Question:** Can one saved model representation support prior and posterior prediction without an agent reimplementing its forward equation?

**Why this matters:** This is a direct way for model-as-data to eliminate a concrete problem observed in the hosted pilot.

**Candidate setup:**

- Reuse a supported model artifact for fitting and the declared predictive operations.
- Specify the prediction target: existing units/groups, new covariates, or new groups are not interchangeable tasks.
- Make observed data versus design-only inputs explicit.
- Inspect all authored helpers to distinguish generic array conversion/plotting from duplicated model mathematics.
- Validate predictive behavior against known properties or an independent reference rather than merely checking that files exist.

**Candidate measurements:** Correct shapes and conditioning, valid predictive distributions, helper code required, duplicated model expressions, and consistency after a model change.

**Candidate failure signals:** Handwritten duplication of the likelihood's forward equation; stale hardcoded expressions after revision; incorrect conditioning; or an undeclared backend switch.

**Baseline:** NumPyro's `Predictive` already provided this advantage in the hosted pilot. Matching that basic capability would close a gap; it would not alone establish superiority.

#### Test 4 — Relocated replay

**Question:** What exactly can a recipient reconstruct from a moved analysis bundle, without the original workspace or conversation?

**Why this matters:** Prior handoff reconstructed summaries from saved draws but did not test whole-analysis replay after relocation.

**Define the guarantee before the test:**

- **Artifact verification:** confirm the specified model/data/result consistency relationships.
- **Summary reconstruction:** recompute reported quantities from saved draws without inference.
- **Computational replay:** rerun the declared computation with an identified engine/environment.
- **Bitwise replay:** reproduce exact bytes under specified supported conditions.
- **Statistical reproducibility:** obtain results satisfying declared tolerances when exact draw identity is not promised.

These are different claims. Decide which are required and which are explicitly out of scope.

**Candidate setup:** Copy a self-contained bundle to a fresh path or machine, remove access to the original checkout and caches as appropriate, and ask an independent process or fresh agent to perform the selected replay operations.

**Candidate measurements:** Missing dependencies, broken absolute paths, required undocumented knowledge, verification results, reconstructed summaries, and replay discrepancies under the declared guarantee.

**Candidate failure signals:** Reliance on original source paths or undeclared state; inability to identify model/data/engine/settings; silent use of changed inputs; or a result claimed stronger than the evidence supports.

**Boundary:** Fingerprints are not signatures or attestations. Say exactly which bytes and relationships are covered, including whether posterior outputs themselves have recorded digests.

#### Test 5 — Revision integrity

**Question:** When the scientific model or data changes, can the workflow prevent stale results from being presented as current?

**Why this matters:** All earlier arms handled a cooperative, scripted prior change. The next test should examine what the artifact design catches, rather than merely whether an agent follows preservation instructions.

**Candidate setup:**

- Predeclare controlled mutations: a prior change, a data change, a model/data mix-up, or an attempt to reuse incompatible saved results.
- Separately consider a harmless source/docstring edit that leaves canonical model IR unchanged; define the expected distinction between source identity and model identity.
- Identify which checks, fits, predictions, and reports depend on each changed input.
- Preserve the original evidence and perform mutations only in test copies.

**Candidate measurements:** Detection of stale or mismatched artifacts, quality of the explanation and repair instruction, correct supersession, preservation of earlier results, and false invalidation of unaffected artifacts.

**Candidate failure signals:** Silent reuse of incompatible results; a report presenting old results as belonging to new inputs; missing provenance linkage; or reliance solely on a cooperative agent remembering the change.

**Baseline:** Let a reasonable NumPyro-plus-manifest workflow attempt the same task. The relevant benefit is preventing a concrete error or reducing effort, not simply having more files or hashes.

**Boundary:** Detecting accidental inconsistency is not the same as resisting an adversary who can rewrite every artifact and manifest. The threat model must be explicit.

### Open decisions before writing a protocol

1. **Which promise is the next priority?** Portable deployment, agent repair, predictive model reuse, relocated replay, or stale-result prevention?
2. **Which current artifacts are under test?** Pin exact repository revisions, binary versions, and supported platform(s); inspect the Rust repository before assuming capabilities.
3. **How much of the endgame is included?** Native CLI only, browser/WASM, visualization, or full human-led study workflow? Avoid labeling a narrower success as the whole vision.
4. **What counts as “one small binary”?** Define runtime prerequisites, model-authoring needs, visualization/export dependencies, network access, and delivery costs.
5. **What is the execution/security boundary?** Distinguish authoring, decoding, numerical execution, CLI file I/O, browser hosting, resource limits, and any server-side computation.
6. **Which replay guarantee is intended?** Decide this before measuring reproducibility.
7. **What would count as failure or change our minds?** Write the rejection criteria before implementation agents begin.
8. **What is the fair baseline?** Use NumPyro where it addresses the same user job, and label capability checks that are not comparative benchmarks.
9. **What is the fixed budget and repetition plan?** Agent comparisons need more than a preferred successful anecdote; cold-start timings need controlled conditions.
10. **Will we test the current product or build a missing capability first?** Keep those activities and their evidence separate.
11. **When will the human-facing vision be tested?** None of these engineering checks alone measures whether a human can understand, challenge, and approve an agent's scientific decisions. A later human-led study remains a distinct evaluation.

A possible discussion starting point is **portable execution first, followed by predictive model reuse**, because the first addresses the major untested differentiator and the second addresses a directly observed weakness. Agent-repair and revision/replay tests can then evaluate the interfaces and guarantees actually available. This ordering is a proposal, not a commitment.

## Appendix C: complexity assessment (2026-09-09)

Record of the exchange that produced this revision.

**User:** "I think I am now convinced that my whole approach to this thing was
well intended but albeit too complex. I am thinking about an MVP for testing
the vision." Then: "can we consolidate it all within one markdown? e.g. if you
are not happy with the next steps of the current one, then we re-write it?
what would you adapt?"

**Assistant's assessment, in summary:**

- Agreed that the approach was too complex, with the refinement that the
  complexity lives in the Python layers and in the experiment design, not in
  the engine. The Python packages total roughly 59k lines across five
  distributions; the portable-execution protocol had 282 lines of pre-run gates.
- The Bayesite executable was inspected directly for the first time in these
  discussions: 1.2 MB, about 4 ms cold start to a typed JSON error, five verbs
  present, `posterior-predictive` missing, no version command.
- The clinic model IR is 4,170 bytes of nested node JSON for seven parameters.
  Whether an agent can author that directly from a one-page card is the open
  question, and the vision permits either direct IR or an eDSL producer.
- The vision reduces to one irreducible claim: an agent with only the binary
  can run the full workflow loop by writing the model as data. Everything else
  is downstream.
- Adaptations made to the previous next-step agenda:
  - Tests 1, 2, and 3 collapsed into one agent session design (the MVP).
  - Tests 4 and 5 moved to the backlog with their questions intact.
  - The scratch-container, telemetry, cold-start-repetition, and event-log
    rigor replaced by a restricted `PATH`, one manifest, and a frozen
    evaluator script.
  - The analytic fixture kept as the engine test for the new verb rather than
    as an agent task.
  - Decision rules written before execution, including what a pass does and
    does not license.
  - Prerequisites named explicitly: `posterior-predictive` verb, a `version`
    verb, the authoring card, and a checksummed release binary.
- The `experiments/portable-execution-pilot/` directory was removed in this
  revision because its content is folded in here. Its full text is at git
  commit `7d29faf`.

**Follow-up in the same session.** The user asked to run the MVP with several
agents: "gpt-6, gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, and you here as
fable 5.1 through claude code". Availability was checked: all four OpenAI
models are listed by Pi under `openai-codex` (`gpt-6` resolves as
`gpt-6-astra`), and Claude Code 2.1.267 is installed with an empty user-level
`CLAUDE.md`. The [Agents](#agents) section, session counts, run order,
and rule R1 were revised accordingly. Two confounds were recorded: harness
differs with model for the Fable arm, and the Fable arm's model family wrote
the card and protocol.
