# bayescycle

`bayescycle` is the Python workflow CLI that connects the `bayeswire` authoring
eDSL to backend engine runs and owns the neutral run-directory contract.

It is intentionally glue:

```text
model.py -> bayeswire ModelMeta -> bayeswire IR JSON -> bayesite engine -> run/
```

## Zero-install quickstart

`bayescycle` is the only thing you install by hand; it provisions the rest.
It is published on PyPI at a lockstep version alongside `bayeswire`,
`bayesjax`, `bayesite-viz`, and `bayesite-idata`.

```bash
uv tool install bayescycle
bayescycle sample model.py --data data.json -o run/
bayescycle plot trace run/
```

`uv tool install bayescycle` installs only `bayescycle` and `bayeswire`
(stdlib-only) — nothing else heavy. In-process JAX sampling is opt-in via
the `[inproc]` extra, which pulls in `bayesjax` (JAX/BlackJAX):

```bash
uv tool install 'bayescycle[inproc]'
```

For joint development across packages, clone this monorepo and use the uv
workspace (`uv sync --package bayescycle [--extra inproc]`) instead of
installing from PyPI; see the root [`AGENTS.md`](../../AGENTS.md).

The `sample` command downloads and sha256-verifies the pinned Bayesite engine
release into `~/.cache/bayescycle/engines/` on first use if it isn't already
on `PATH` (see "Auto-provisioning" below). The `plot` command reaches
`bayesite-viz` the same way an agent invoking it from the command line would:
via `uvx` against a pinned PyPI version, so there is nothing to `pip install`
for visualization either. Both steps print a one-line notice to stderr the
first time they fetch something; subsequent runs are cache hits.

## Package split

- **bayeswire**: Python model declaration eDSL, the `bayeswire_ir` wire format,
  its normative spec, and the conformance corpus. Stdlib only.
- **bayesite**: Rust engine and IR-level CLI. It accepts IR plus data and writes
  draws. It stays dependency-free and WASM-clean.
- **bayesjax**: JAX/BlackJAX sampling backend for bayeswire models. Optional
  here; installed via the `[inproc]` extra.
- **bayescycle**: Python workflow harness. It executes a Python model file,
  serializes IR, invokes a backend, and owns run-directory ergonomics and the
  run artifact contract.

## Trust surface of the default path

With the default `--backend bayesite`, running a workflow executes `model.py`
in a Python environment containing exactly one stdlib-only package
(`bayeswire`, plus `bayescycle` itself) and invokes one auditable
zero-dependency Rust binary. No JAX on the default path, ever. That is the
honest version of the original zero-dependency claim: not "no supply-chain
danger", but every step of the default path is small enough to audit, pinned
enough to reproduce, and deterministic enough to replay.

## Initial CLI

```bash
bayescycle sample model.py --data data.json -o run/
```

By default this prepares:

```text
run/model.ir.json
run/data.json       # canonical bayescycle.data.json.v1 snapshot
run/manifest.json   # artifact format manifest
run/dims.json       # optional; written only when the model declares dimension metadata
```

and invokes the Bayesite adapter, which passes the canonical `run/data.json`
straight to the engine (the engine parses the `bayescycle.data.json.v1` wrapper
natively, so both backends fingerprint the same bytes; see
`docs/posterior-draws-v0.md`):

```bash
bayesite sample --model run/model.ir.json --data run/data.json --out run/posterior.ndjson
```

The engine is asked to write draws to:

```text
run/posterior.ndjson
```

Use `--show-plan` to print the planned backend action without creating or rewriting
run-directory artifacts. Plan output records both the selected backend and the
integration mode: Bayesite is an `external-command` adapter, while bayesjax is
an `in-process-python` adapter.

If a Python file declares independent root models, choose one explicitly:

```bash
bayescycle sample model.py --model LogisticRegression --data data.json -o run/
```

Models referenced through `Submodel(...)` are components rather than competing
roots. When a file contains one unreferenced root plus its locally declared
components, bayescycle selects that root automatically; `--model` can still
select a component explicitly.

Common sampler settings are first-class workflow options:

```bash
bayescycle sample model.py --data data.json -o run/ --seed 123 --chains 2 --warmup 100 --draws 100
```

The default backend invokes the Bayesite executable. An in-process JAX backend is
available when the optional dependencies are installed:

```bash
bayescycle sample model.py --data data.json -o run/ --backend bayesjax \
  --seed 123 --chains 2 --warmup 100 --draws 100 --max-treedepth 10 --target-accept 0.8
```

`--engine /path/to/bayesite` remains the Bayesite executable override and only
applies to `--backend bayesite`. Before execution, Bayescycle preflights the
selected Bayesite binary for existence, executability, and required subcommand
support so stale engines fail before run-directory writes. Install the
in-process dependencies with `bayescycle[inproc]`.

### Auto-provisioning

When `--engine` is unset and no `bayesite` executable is found on `PATH`,
Bayesite-backed commands (`sample`, `prior-predictive`, `simulate`, `recover`,
`sbc`, `diagnose`, `posterior-predictive`, `posterior-check`,
`recover-check`, `replay`, `idata`, `plot`) auto-download the pinned Bayesite
engine release into the bayescycle cache before running. Opt out with
`--no-auto-provision` on any of those commands, or set
`BAYESCYCLE_NO_AUTO_PROVISION=1` for the whole environment (the flag wins if
both are given). With auto-provisioning disabled, an unresolved engine fails
with a clear error pointing at `bayescycle engine ensure` or `--engine`.

Additional engine flags can be forwarded after `--`. The engine `--out` flag is reserved
so the run directory always contains `run/posterior.ndjson`.

```bash
bayescycle sample model.py --data data.json -o run/ -- --experimental-engine-flag
```

The run-directory contract is documented in [`docs/run-directory-v0.md`](docs/run-directory-v0.md),
[`docs/canonical-data-artifacts.md`](docs/canonical-data-artifacts.md), and
[`docs/posterior-draws-v0.md`](docs/posterior-draws-v0.md). Bayesite is the
initial backend that emits this contract, but the contract is owned by
`bayescycle` rather than by a specific sampler.

Simulation-gate commands are also first-class and own their run-directory output paths:

```bash
bayescycle prior-predictive model.py --data inputs.json -o runs/prior-0001 --seed 123 --draws 500
bayescycle simulate model.py --data inputs.json --truth truth.json -o runs/sim-0001 --seed 1
bayescycle recover model.py --scenario scenario.json -o runs/recover-0001
bayescycle sbc model.py --scenario scenario.json -o runs/sbc-0001 --replicates 100
```

Run directories are append-only discovery records. Commands refuse to reuse a
non-empty output directory or overwrite an existing follow-up artifact; use a new
run id for a new attempt. Fresh model-level runs also write `run.json` with
narrow provenance metadata for future indexing and replay.

Audit a completed model-level run by replaying it into a fresh directory:

```bash
bayescycle replay run/ -o run-replay/
bayescycle replay run/ -o run-replay/ --check-only
```

Replay verifies the recorded model/input source hashes before executing, refuses
if sources drifted, and prints a per-artifact byte comparison. Use `--engine` on
`replay` when replaying a Bayesite run with a non-default executable.

`prior-predictive` supports both backends. `simulate`, `recover`, and `sbc` are
currently Bayesite-backed; selecting `--backend bayesjax` returns a clear
unsupported-profile error rather than falling back silently.

`simulate` writes `runs/sim-0001/simulated_data.json` as canonical
`bayescycle.data.json.v1`, so a downstream sample command can use it with the
same backend or with `--backend bayesjax` through the adapter boundary. For
multi-stage intent, validate the backend plan before creating run directories:

```bash
bayescycle workflow-plan --backend bayesite
bayescycle workflow-plan --config workflow.toml
```

Run-directory follow-up phases can be orchestrated without repeating owned paths:

```bash
bayescycle diagnose run/
bayescycle posterior-predictive run/ --seed 456
bayescycle posterior-check run/ --seed 456
bayescycle recover-check run/ --truth truth.json --targets targets.json --interval 0.8
```

These invoke Bayesite with `run/posterior.ndjson` as the fit input and write:

```text
run/diagnostics.json
run/posterior_predictive.ndjson
run/posterior_check.json
run/recovery_check.json
```

## Engine management

The Bayesite engine binary is normally provisioned automatically (see
"Auto-provisioning" above), but the pinned release can also be managed
directly:

```bash
bayescycle engine ensure              # provision the pinned release if missing; prints its path
bayescycle engine ensure --force      # re-download and re-verify even if already cached
bayescycle engine path                # print the resolved path (PATH, else the cache); no network
bayescycle engine info                # print the engine's structured capabilities as JSON
```

`ensure` and `path` accept `--cache-root` to target a non-default cache
directory; `ensure` accepts `--base-url` to fetch from a mirrored or
airgapped release host; `info` accepts `--engine` to inspect a specific
binary instead of the resolved default. The pinned version and per-target
checksums live in `PINNED_ENGINE_RELEASE` in
`src/bayescycle/backends/bayesite/provisioning.py`, bumped with
`uv run python scripts/bump_engine_release.py --tag vX.Y.Z`.

## Visualization: idata and plot

`bayescycle idata` exports a run directory to an ArviZ-compatible NetCDF fit
file via `bayesite-viz`'s `bayesite-idata`, and `bayescycle plot <verb>`
renders one of its plots from that fit file. Both run `bayesite-viz` through
`uvx` against a pinned PyPI version (`BAYESITE_VIZ_SOURCE` in
`src/bayescycle/backends/bayesite_viz/uvx_runner.py`); no separate `pip
install` of `bayesite-viz` is required, only `uv`/`uvx` on `PATH`.

```bash
bayescycle idata run/                       # writes run/fit.nc (default output path)
bayescycle idata run/ -o run/custom-fit.nc
bayescycle idata run/ --validate require    # require|warn|skip; forwarded to bayesite-idata
```

```bash
bayescycle plot trace run/
bayescycle plot posterior run/ --kind hist -o run/posterior.png
bayescycle plot ppc run/ --kind cumulative --svg
```

The nine supported verbs are `trace`, `rank`, `forest`, `energies`, `pair`,
`posterior`, `autocorr`, `ess-rhat`, and `ppc`. `--kind` only applies to
`posterior` and `ppc`. `--var` (repeatable), `--coords key=value`
(repeatable), `-b/--backend` (`matplotlib`, `bokeh`, or `plotly`), `-f/--format`
(stdout announce format), and `--svg` forward straight to `bayesite-viz`.

`plot` auto-runs `idata` when `run/fit.nc` (or `--fit`) doesn't exist yet, so
`bayescycle plot trace run/` works directly after `sample` without a separate
`idata` step. Pass `--no-auto-idata` to require an existing fit file instead.
Both commands accept `--engine` and `--no-auto-provision` /
`BAYESCYCLE_NO_AUTO_PROVISION` the same way the other Bayesite-backed
commands do (see "Auto-provisioning" above), since an auto-run `idata` step
still needs a Bayesite engine.

### Deterministic uvx environments and offline pre-warming

`idata` and `plot` pin bayesite-viz to an exact PyPI version
(`BAYESITE_VIZ_SOURCE`), but that alone doesn't pin *its* dependencies
(arviz, matplotlib, netcdf4, xarray, ...): `uvx` resolves those as `>=`
ranges at invocation time, so an upstream release could otherwise change
what gets installed with no change on our side. Every `uvx` invocation also
passes `--exclude-newer` (`BAYESITE_VIZ_EXCLUDE_NEWER` in the same module),
so dependency resolution is a pure function of the two pins together and the
plot/idata environments are reproducible.

```bash
bayescycle warmup   # pre-materializes both uvx environments (bayesite-idata, bayesite-viz)
```

Run `bayescycle warmup` once while online to populate `uvx`'s cache; `idata`
and `plot` then work offline and fail fast at warmup time instead of
mid-workflow if the environments can't be resolved.

## Agentic study workflow

The optional `.agents/skills/bayescycle-study/` skill defines a lightweight,
human-gated Bayesian study protocol. It keeps canonical study state in JSON,
records proposed state transitions as patches, treats Bayesite NDJSON as
telemetry rather than scientific state, uses DAGs as scaffolding for causal
generative models, and makes visualization a required phase-gate concern through
`bayesite-viz` artifacts.

The skill is intentionally outside `src/bayescycle`: it may use the CLI, but the
Python package remains a narrow deterministic workflow harness. Its protocol is
maintained in [`.agents/skills/bayescycle-study/`](../../.agents/skills/bayescycle-study/).

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

## Boundary invariant

`bayescycle` may execute Python authoring code and depends on `bayeswire` for
authoring semantics and IR serialization. It may serialize authoring-side
metadata that `bayeswire` explicitly exposes into the run directory, such as
optional dimension labels in `dims.json`, and sampler facts explicitly exposed
by the selected backend. It must not invent model semantics, infer labels from
shapes/names, contain inference algorithms, distribution math, IR evaluation,
or sampler logic. Authoring semantics belong to `bayeswire`; sampler facts
belong to `bayesjax` or Bayesite.
