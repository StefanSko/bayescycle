# Local-model replication — registered before study agents run

## Hypothesis

Does the narrower Bayeswire/Bayesjax interface, or its Bayescycle CLI, help the
installed smaller local agent conduct this Bayesian analysis more reliably than
NumPyro? This is a replication of the hosted-model fixture, not a new inference
algorithm or harness implementation.

## Fixed local agent

Use the already configured `ollama/gemma4-pi:12b`: Gemma 4 11.9B, Q4_K_M,
32,768-token configured context, 4,096-token maximum response. Existing Ollama
parameters: temperature 1, top_k 64, top_p 0.95. Pi requests medium thinking;
the installed OpenAI-compatible provider disables reasoning_effort support, so
this is NOT a claim of equal reasoning budget to the hosted model. No model
download, quantization change, context enlargement, custom scaffold or global
configuration modification is permitted during the trial.

The machine has 24 GiB unified memory and 12 logical CPUs. All model requests
use the configured loopback Ollama endpoint. Scientific Python uses the previous
pilot's locked environment (JAX 0.10.2, BlackJAX 1.5, NumPyro 0.21.0, ArviZ 0.22).
Root/provider/version metadata is recorded separately by the evaluator.

## Arms and unchanged scientific fixture

- A: NumPyro + ArviZ + ordinary scripts.
- B: Bayeswire + direct public Bayesjax APIs + ArviZ.
- C: Bayeswire + Bayescycle CLI, explicitly Bayesjax for fitting. Native artifacts
  remain required; supplementary unsupported operations must be identified.

All arms use Pi's ordinary read/write/edit/bash tools. This tests usability of
existing scientific APIs, not a newly constrained tool schema, sandbox or enforced
permission system. No Rust fallback, MCP server or new harness integration.

Eight clinics, 20 observations each. Use the byte-identical main and separate
recovery fixtures from the hosted pilot, copied into each workspace. Main truth
and hosted solutions/results are evaluator-only; local agents must not inspect
any earlier pilot report, code, figure, session or numerical result. A common
scientific environment is not permission to read its neighboring study outputs.

Model, with non-centered clinic intercepts:

- alpha ~ Normal(0, 2)
- beta ~ Normal(0, 1)
- tau ~ HalfNormal(1)
- z_j ~ Normal(0, 1), j=0,...,7
- sigma ~ HalfNormal(1)
- y_i ~ Normal(alpha + tau * (clinic_design @ z)_i + beta*x_i, sigma)

Estimand: within-clinic expected outcome difference for x -> x+1, beta.
Associational, not an identified causal effect. No missingness. Treat design as fixed.
Use float64; 4 chains; 500 warmup and 1000 retained draws per chain; target
acceptance 0.9; max tree depth 10. Seeds: 4200 for 200 prior predictions, 4301 for
recovery fitting, 4201 for main fitting, 4202 for 200 main posterior predictions.
Do not alter scientific settings, hunt for seeds or silently change backends.

## Procedure and budgets

1. A read-only, non-scientific tool-calling smoke test, separately logged, verifies
   that the installed Ollama/Pi integration can perform a tool call. Failure here
   is an integration blocker, not evidence against a Bayesian package.
2. Execute initial A/B/C sessions sequentially in a seeded shuffled order, with
   fresh workspaces/context. Each has the same 15-minute initial budget as the
   hosted pilot. Budget includes reasoning, code, tools and analysis; slow local
   decoding can cause noncompletion without demonstrating an API flaw.
3. If an arm has genuinely completed the initial fixture, snapshot its original
   source/results and request the same beta-prior revision: Normal(0,0.25), in a
   fresh session (10-minute budget). Preserve original artifacts.
4. Only for arms with completed revised work, perform the same file-only fresh
   handoff, no refitting (5 minutes).
5. Partial/failed arms retain all evidence. Do not repair their code with the
   hosted evaluator, secretly supply solved templates, or declare missing stages
   passed. Ordinary self-repairs within the budget are allowed and counted.

No agent may fabricate human approval. Actual human understanding, trust and
scientific decision quality remain unmeasured; the full gated study skill is not
invoked. This evaluates engineering capability, not the finished product idea.

## Outputs and assessment

Use the same output paths/schema as the hosted fixture: separate initial and
recovery posterior NPZs (alpha/beta/tau/z/sigma, chain-by-draw), summaries with
beta mean/SD/95% equal-tailed interval/MCSE/rank Rhat/bulk and tail ESS, native
divergences/settings, predictive figures and STUDY.md. Revised outputs and any
handoff must remain separate and explicitly proposed, never human-approved.

Primary endpoints: completion of the specified task within budget, equivalence of
model/settings to the fixture, and correctly interpreted numerical diagnostics.
Check shapes/finiteness/positive scales; recompute ArviZ rank Rhat <=1.01, bulk and
tail ESS >=400, and zero retained divergences across parameter coordinates.
For equivalent models compare main beta means to the matching hosted-arm
reference using 4*hypot(local MCSE, reference MCSE). Posterior agreement alone
cannot establish that the implemented model was correct. One recovery interval's
truth coverage is descriptive, not a calibration verdict.

Record failed operations, self-repairs, fabricated success/approval claims,
API misunderstandings, source/result preservation and handoff. Separately identify
provider/tool-parser/context/output-budget failures. Record actual wall time and
provider token telemetry where available; zero API price is not zero hardware cost.

One attempt per arm, one model, one fixture: this is a qualitative screening test,
not an intrinsic error-rate estimate. The LLM has its existing stochastic decoding
parameters and no claim of seed-identical text generation. Sequential local runs
are not directly speed-comparable to the hosted pilot's concurrent runs. A finding
that all arms fail would motivate a smaller capability test, not establish that
all scientific libraries are unusable or that more infrastructure is needed.
