# Agentic Bayesian vision: experiment retrospective and next tests

This document preserves the discussion about the experiments in `experiments/`, the project's intended vision, and the proposed next tests. The conversation is reproduced in full below, followed by a planning aid for continuing the discussion.

**Status:** discussion record and proposed test agenda. This is not an approved implementation plan, a registered experiment protocol, or a decision to change package responsibilities. No new experiments were run during this discussion.

## Contents

1. [Evidence consulted](#evidence-consulted)
2. [Full conversation](#full-conversation)
3. [Where the discussion leaves us](#where-the-discussion-leaves-us)
4. [Next-test discussion draft](#next-test-discussion-draft)
5. [Open decisions before writing a protocol](#open-decisions-before-writing-a-protocol)

## Evidence consulted

The assistant inspected the experiment directory and read these files during the discussion:

- [Hosted pilot README](workflow-value-pilot/README.md)
- [Hosted pilot evaluation](workflow-value-pilot/evaluator/evaluation.json)
- [Hosted pilot protocol](workflow-value-pilot/PROTOCOL.md)
- [Hosted pilot initial brief](workflow-value-pilot/INITIAL-BRIEF.md)
- [Local Gemma pilot README](gemma-workflow-pilot/README.md)
- [Local Gemma pilot assessment](gemma-workflow-pilot/evaluator/assessment.json)
- [NumPyro authoring API card](gemma-workflow-pilot/a-model/API.md)
- [Bayeswire authoring API card](gemma-workflow-pilot/b-model/API.md)

Saved walkthroughs:

- [Hosted pilot HTML report](workflow-value-pilot/report.html)
- [Local Gemma pilot HTML report](gemma-workflow-pilot/report.html)

The assessments below concern those saved experiments. The Rust engine, browser deployment, and security architecture were not independently audited during this discussion. Descriptions of the endgame are aspirations, not assertions that every feature currently exists.

## Full conversation

### 1. User: concern about the experiments

can we discuss about the experiments we have conducted in experiments/. it is was supposed to show some usefulness of this library for agentic bayesian workflow, but seems it failed and numpyro seemed more succesful.

### 2. Assistant: initial reading of the evidence

Yes. I read both experiment summaries, their evaluation results, and the task/API briefs. **Your interpretation is broadly right: NumPyro was the stronger practical choice in these trials.** I would not explain that away—but I would distinguish which hypotheses failed.

#### What the evidence actually says

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

#### Why I think this happened

##### 1. A smaller language is not necessarily easier for an agent

NumPyro benefits from familiar Python, standard array operations, and likely much greater representation in training data.

Bayeswire introduces different concepts: class declarations, explicit constraints, symbolic data declarations, and `linear(...).apply(...)` instead of `@`. Those might offer useful validation properties, but the agent must learn them first.

**Restricting expressiveness can make validation easier without making authoring easier.** We appear to have treated those as more closely connected than these results support.

Familiarity is a possible experimental confound—but it is also a real product advantage.

##### 2. The workflow layer did not absorb enough complexity

The clearest concrete problem is posterior prediction: B and C duplicated the model’s forward equation in supplementary scripts, whereas A used NumPyro’s `Predictive`.

That is more than inconvenient. It creates another place where the implemented science can drift.

C also had to navigate backend-dependent command availability and artifact conversions. Explicit boundaries may be good engineering internally, but **if agents have to manage all those boundaries themselves, the user-facing workflow is not simpler.**

##### 3. Ordinary scripts were a stronger baseline than expected

The NumPyro agent preserved models, saved draws, recorded hashes, explained revisions, and left a usable handoff.

Bayescycle supplied more standardized provenance, which is real value. But the experiment did not show that those extra guarantees prevented a mistake or enabled something the simpler approach could not do.

“We have structured artifacts” is a capability. “Those artifacts materially improve research work” is a separate claim.

#### What remains open—and what should not become an excuse

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

### 3. User: the intended vision

my vision is this one:
You're building the first Bayesian modeling system designed for the agentic era — where a model is data, not compiled code. A readable Python eDSL (or an agent, or any conformant producer) emits a versioned, language-neutral IR; that IR is executed by interchangeable engines: a JAX reference backend that defines the semantics and handles large models, and bayesite — a zero-dependency, capability-free Rust engine, sqlite-small, that starts in milliseconds and runs anywhere: a terminal, an agent's ephemeral sandbox, CI, or a browser tab via WASM. Around the engines sits a full workflow discipline as CLI verbs — prior predictive, simulate, recover, SBC, sample, diagnose, posterior predictive — with run directories, fingerprinted provenance binding every fit to the exact bytes of its model and data, and replayable analyses.

The three convictions underneath it:

No compile step, ever. Every incumbent — Stan, nutpie, even browser-Stan — compiles per model, which forces toolchains and servers into the loop. Model-as-IR fed to one pre-built engine deletes that entire category of friction, and it's the property that makes agent iteration (dozens of model variants in a sandbox) and true client-side execution possible at all.
Agent-operability as the primary design constraint. Small in-context grammar, typed prescriptive errors that work as repair instructions, stateless verbs with exit codes, filesystem artifacts. The tool's first user is an agent running the workflow loop; humans get the readable surface and the documents (Showboat-style narrated, re-runnable analyses) that fall out of it.
Safety by architecture, not mitigation. IR decodes without executing code; sampling is capability-free; the only untrusted-code exposure is one small authoring step that can live in the user's own browser. In the eventual deployed product — an agent drafts the model server-side as data, the user adjusts it in readable Python, everything executes client-side — your server runs no untrusted computation. The risk isn't contained; it's designed out.

Holding it together is the contract-first governance you've now made concrete in bayescycle: bayeswire (stdlib-only eDSL + IR + specs + conformance corpus) as the slow-moving center, bayesjax as the oracle every engine must statistically agree with, bayescycle as the lean workflow CLI, viz and export tools kept heavy but firewalled behind process boundaries — and reproducibility (hashable IR, diffable model iterations, fingerprinted fits) as a side effect of the artifact design rather than a feature bolted on.

The endgame: anyone — agent or human — can go from "describe a modeling problem" to a fitted, diagnosed, visualized, reproducible Bayesian analysis, anywhere code can run, with nothing to install but one small binary. The sqlite of Bayesian workflow, built for a world where the analyst is often an agent.

### 4. Assistant: revised interpretation in light of the vision

That clarifies something important. **Your vision is not primarily “a nicer NumPyro.” It is a portable execution contract for Bayesian models, with an agent-operable workflow around it.** My earlier suggestion to default to NumPyro addressed the analysis task, but not that larger objective.

I think the vision is coherent. I also think the experiments revealed a genuine weakness—not a refutation of its central architecture.

#### The experiments tested the cost, but largely not the distinctive benefit

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

#### Three claims I would sharpen

##### “No compile step, ever”

The defensible promise is:

> **No model-specific compilation or user toolchain required by the portable engine.**

NumPyro does not require a Stan-style external compilation workflow, although JAX normally performs runtime compilation. Your JAX backend also makes “no compile step, ever” too broad.

Likewise, eliminating model-specific compilation is an excellent route to browser execution—not the only technically possible route.

The narrower statement is already compelling. It does not need the universal claim about incumbents.

##### “Safety by architecture”

This is a substantial differentiator **if the execution boundary genuinely accepts only validated data and exposes no ambient capabilities**.

But distinguish:

- IR decoding from Python authoring: arbitrary Python remains code execution.
- Capability isolation from resource safety: a model can still exhaust time or memory.
- Architectural restrictions from implementation correctness: parsers, numerical kernels, and WASM hosts still need scrutiny.

You can design out *the requirement to execute submitted arbitrary code on your server*. That is powerful. “The risk is designed out” overstates it.

##### “JAX defines the semantics”

I would put the authority in the **specification**, with JAX as an executable reference and numerical oracle.

Otherwise an implementation quirk risks becoming the contract. Engines should agree on model meaning and target density; finite posterior samples should satisfy appropriate statistical checks, not necessarily match.

#### The uncomfortable finding still stands

Your architecture can be right for portability while the interface is wrong for agents.

The compact Gemma probe suggests that a small unfamiliar grammar is not automatically easier than a larger familiar one. The hosted trial showed that workflow boundaries currently leak into the agent’s work.

Most tellingly, **an agent had to rewrite the forward equation for posterior prediction despite the system already possessing the model as data**.

That is exactly where your architecture ought to help. The IR should allow supported predictive operations to reuse the model, rather than requiring another implementation of its mathematics.

This is not an argument against model-as-data. It is evidence that its benefits are not yet fully delivered.

#### What I would test next

Not another generic “which library fits this regression faster?” contest. Test the distinctive promises directly, while retaining NumPyro as an honest baseline where applicable:

1. **Portable execution:** hand a clean environment only the binary, IR, and data. Complete a supported workflow without Python, dependency resolution, or model compilation.
2. **Agent repair:** give agents invalid models and measure whether typed errors lead to correct repairs within fixed budgets. Compare direct IR production with eDSL authoring.
3. **Model reuse:** execute prior and posterior prediction without duplicated model equations.
4. **Relocated replay:** move an analysis to a fresh location or machine; verify its inputs and reconstruct results under an explicitly defined replay guarantee.
5. **Revision integrity:** change model or data bytes and test whether stale results are identified rather than silently reused.

These should have declared failure criteria—not be demonstrations arranged to guarantee a win.

**My revised position: keep the vision, but narrow the next proof.** Show that the same model artifact can be safely transported, executed, inspected, and reused with materially less environmental machinery. Then show that agents can operate that system reliably.

The “SQLite of Bayesian workflow” claim becomes convincing when the user experiences that simplicity—not merely when the repository has the architecture to support it.

### 5. User: request to preserve the discussion

can you put all we discussed into a markdown, wihtout any loss of context, so we can disuss thorugh your next suggested test steps?

## Where the discussion leaves us

This section summarizes the discussion; it does not replace the full conversation or imply that the user has approved the assistant's recommendations.

### Observations we must preserve

- NumPyro was the lower-friction solution in the hosted pilot. All arms completed the prescribed numerical task, revision, and handoff.
- Direct Bayesjax and Bayescycle worked on that fixture; the negative result concerns demonstrated usability advantage, not evidence of incorrect inference.
- Bayescycle supplied standardized native artifacts and fingerprints, but ordinary scripts also passed the tested preservation and handoff requirements.
- The full local-agent attempts failed before scientific API use. Those failures cannot isolate backend or modeling-language quality.
- The compact local authoring probe succeeded only with NumPyro. It is an exploratory, single-attempt result with budget, presentation, familiarity, and stochastic-decoding limitations.
- B and C duplicated the forward equation for posterior prediction. This is a concrete workflow usability gap.
- The Rust engine, deployment simplicity, browser execution, exact relocated replay, and actual human approval workflow were not tested by these pilots.

### Additional qualifications retained from the underlying reports

These details were in the evidence read during the discussion and remain important when interpreting “successful numerical results”:

- All 12 completed saved fits passed the specified chain-diagnostic thresholds with zero retained-draw divergences. This does not establish scientific validity or calibration.
- All three initial recovery intervals missed the known beta of 0.4; the tighter-prior revision did not fix that. Agents reported the miss. One recovery dataset is not a calibration test.
- B and C used the same engine/settings and produced bit-identical posterior arrays. Their agreement is not independent cross-engine corroboration.
- Hosted implementation sessions ran concurrently with shared hardware/cache effects. Session times are not sampler benchmarks.
- The compact local NumPyro model was fitted and numerically validated by the evaluator, not by the local authoring agent.
- Local hashes establish specified byte-consistency relationships, not signatures, authenticated execution, or proof that arbitrary source produced a particular posterior.
- The full gated study skill was deliberately not executed. No actual human approvals were measured or invented.

### The important change in interpretation

Before the vision was clarified, the assistant recommended treating NumPyro as the default and questioning the need for the custom computational stack.

After clarification, the assistant distinguished two objectives:

1. **Completing Bayesian analyses efficiently in an already-installed Python environment.** The existing NumPyro baseline performed better on the tested task.
2. **Transporting and executing Bayesian models as data through a small, portable, constrained engine with standardized workflow artifacts.** The current experiments did not substantially test this distinctive proposition.

The revised recommendation is not to discard the vision or to declare it validated. It is to test its distinctive promises directly while preserving the unfavorable evidence about present authoring and workflow friction.

### Claims versus aspirations

- “First,” “sqlite-small,” “starts in milliseconds,” “runs anywhere,” “handles large models,” and “nothing to install but one small binary” need operational definitions or supporting evidence before being treated as established product claims.
- “No model-specific compilation or user toolchain required by the portable engine” is more precise than “no compile step, ever.”
- A normative specification and an executable reference have different roles. Statistical agreement also differs from semantic conformance or identical draws.
- Capability restrictions do not establish unlimited resource safety, parser correctness, or scientific correctness.
- Existing separate Python visualization/export environments are relevant to the eventual single-binary end-to-end claim: the next protocol must say whether visualization is inside or outside the demonstrated scope.

## Next-test discussion draft

**Everything below is a proposed elaboration for the next discussion, not an already agreed protocol.** The five test themes came from the conversation; the questions and candidate measurements below make them easier to scope.

### General approach

Separate deterministic product-capability checks from stochastic agent evaluations:

- First establish what the pinned released or development artifacts can actually do through public interfaces.
- Then measure whether agents can use those capabilities reliably.
- Missing functionality is a recorded result, not permission for an evaluator to silently fill the gap.
- If implementation work is chosen afterward, distinguish the pre-change baseline from the subsequent validation.
- Keep the old experiments and failures intact. Use new directories for new protocols and evidence.

### Test 1 — Portable execution

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

### Test 2 — Agent authoring and repair

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

### Test 3 — Model reuse for prediction

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

### Test 4 — Relocated replay

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

### Test 5 — Revision integrity

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

## Open decisions before writing a protocol

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
