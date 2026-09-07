# Initial implementation task

You are one independent analysis agent in a bounded engineering pilot. Read
`../PROTOCOL.md` completely, then implement and EXECUTE only the initial stage
in your assigned arm. Do not execute the later revision or handoff yet.

You have 15 minutes. Write only within your assigned current working directory.
Do not read other arms, `../evaluator/`, `../prepare.py`, or any other agent's
session/logs. The shared protocol, this brief, repository package docs/public API
source, spec, and installed library documentation may be read. No browsing of
user secrets. Do not modify package source, shared inputs, locks, dependencies,
or repository settings. Use only public APIs (reading public API source is OK).

The scientific model/settings are a FIXED evaluator fixture, not a human-approved
scientific study. Execute the fixture as an engineering test; do not invent human
approvals. Your STUDY.md must explicitly say it awaits human scientific review.
Do not invoke the full gated study skill or claim to have satisfied its gates.
Do not change the model, counts, priors, diagnostic thresholds, or backend to make
results pass. Record failures and limitations. Before the supplied-data fit,
perform prior prediction and separate recovery. No silent fit retries to hunt for
a favorable random seed.

Environment is already installed. For every Python command use:

```
uv run --project .. python your_script.py
```

For the CLI use `uv run --project .. bayescycle ...`. Do not run `uv sync` or
install extra dependencies yourself. The separate pinned `uvx` visualization
boundary is allowed for C if the CLI requires it; record that overhead/failure.
JAX_ENABLE_X64=true, MPLBACKEND=Agg, four host CPU devices and OMP_NUM_THREADS=1
are supplied to the session. All arms use ArviZ 0.22 for independent rank Rhat,
bulk/tail ESS and MCSE; preserve any backend-native diagnostics too.

Output conventions, in addition to the protocol:

- Save `results/initial/posterior.npz` and `results/recovery/posterior.npz`, with
  each parameter under its own name (`alpha`, `beta`, `tau`, `z`, `sigma`), shaped
  (chain, draw, *parameter_shape). Preserve native C posterior artifacts also.
- Save `results/initial/result.json` and `results/recovery/result.json` with
  top-level `beta` containing `mean`, `sd`, `q025`, `q975`, `mcse_mean`, `rhat`,
  `ess_bulk`, `ess_tail`; top-level `divergences` and `sampler` metadata.
  Extra fields are welcome if useful, but keep output compact.
- Save figures and underlying predictive draws; inspect figures with the read
  tool if time permits. Do not pretend a figure was inspected if it was not.
- Write STUDY.md and a runnable analysis script with clear commands. Include
  hashes of the exact input/model files used; do not substitute hashes of later
  reserialized bytes for original input bytes. Keep original inputs unchanged.
- Refuse accidental overwrites of completed result directories. Failed scratch
  attempts may be kept separately; identify them in the notes.
- Finish with a concise summary and unresolved issues. Do not wait for an
  interactive reply; this is a print-mode experiment, not human approval.

For Bayeswire/Bayesjax consult ../../../packages/bayeswire/README.md,
../../../packages/bayesjax/README.md, package invariants and public APIs as needed.
For C also consult ../../../packages/bayescycle/README.md, the spec, and
../../../.agents/skills/bayescycle-study/adapters/bayesjax-inproc-v1.md. Keep
model declaration in Bayeswire (the adapter may use historical ownership names).
No Rust fallback. If the CLI doesn't implement recovery as a dedicated command,
sampling the shared recovery fixture through the CLI is legitimate recovery;
clearly label manual checks or predictive scripts needed to complete the task.
