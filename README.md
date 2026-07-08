# bayescycle

**A Bayesian workflow an agent can run end-to-end through files,
deterministically, with an audit trail.**

Every workflow step is a command that reads files and writes files. Every
run is seeded and replayable. Run directories are append-only. Provenance
records what was actually done. The customer is an agent — and the human
auditing it — and the product is trustworthy *process*, not a sampler.

```text
model.py ──bayeswire──▶ model.ir.json ──bayesite──▶ run/ ──bayesite-idata──▶ fit.nc ──bayesite-viz──▶ plots
   │                        (IR)         (engine)    │                                     ▲
   └── declarative eDSL                              └── diagnostics, posterior checks ────┘
```

## The toolchain

This repository is a uv-workspace monorepo publishing five lockstep Python
distributions, plus one external Rust engine:

| Package | What it is | Trust surface |
|---|---|---|
| [`bayeswire`](packages/bayeswire) | Model declaration eDSL, `bayeswire_ir` wire codec, dimension sidecars, the normative [spec](spec/), and the golden conformance corpus | **Stdlib only** |
| [`bayescycle`](packages/bayescycle) | Workflow CLI: model file → IR → run directory → backend invocation → follow-up phases | bayeswire only |
| [`bayesjax`](packages/bayesjax) | JAX/BlackJAX reference backend and float64 oracle behind the corpus fixtures | JAX stack (opt-in `[inproc]`) |
| [`bayesite-idata`](packages/bayesite-idata) | Run directory → ArviZ DataTree/NetCDF exporter | arviz stack, behind a `uvx` process boundary |
| [`bayesite-viz`](packages/bayesite-viz) | ArviZ plotting CLI (`trace`, `rank`, `ppc`, `ess-rhat`, …) | arviz stack, behind a `uvx` process boundary |
| [bayesite](https://github.com/StefanSko/bayesite) | Zero-dependency Rust NUTS engine (separate repository) | One auditable static binary |

The default execution path — author a model, serialize IR, sample with the
Bayesite engine — runs Python in an environment containing exactly one
stdlib-only package plus one pinned release binary. JAX never enters it;
CI asserts this. The heavyweight visualization stack is reachable only
through a `uvx` process boundary at a pinned version and never joins
`bayescycle`'s dependency closure.

## Why it looks like this

- **The contract is the product.** The tools compose through one normative
  wire spec ([`spec/`](spec/)) and a golden conformance corpus that every
  producer and consumer is tested against — the Rust engine vendors both by
  byte-reviewed file copy and proves logp/gradient parity against the same
  fixtures.
- **The model is data, not code.** `model.ir.json` is an inspectable
  document; decoding runs no user code, so every downstream phase — a
  diagnose, a posterior check, a re-fit months later — proceeds from the
  artifact alone.
- **Determinism over convenience.** Exact-version sibling pins, a pinned
  engine release, `uvx --exclude-newer` for the plotting boundary: fixed
  inputs and seeds give fixed bytes.

## Quickstart (from this repository)

```bash
# Workflow CLI on the default no-JAX path
uv sync --package bayescycle
uv run bayescycle --help

# Provision the pinned Bayesite engine release, then sample
uv run bayescycle engine ensure
uv run bayescycle sample model.py --data data.json -o run/

# Follow-up phases own their run-directory artifacts
uv run bayescycle diagnose run/
uv run bayescycle posterior-check run/ --seed 456

# Export and plot (heavyweight stack stays behind uvx)
uv run bayescycle plot trace run/
```

The in-process JAX backend is opt-in:

```bash
uv sync --package bayescycle --extra inproc
uv run bayescycle sample model.py --data data.json -o run/ --backend bayesjax
```

An executed end-to-end walkthrough with the philosophy behind each phase is
in [`docs/toolchain.html`](docs/toolchain.html).

## Repository layout

```text
packages/    five Python packages (three workspace members; the two viz
             packages are standalone uv projects behind the uvx boundary)
spec/        the normative wire contracts: IR format, tags, data documents,
             dimension sidecars, model/data fingerprint
docs/        toolchain walkthrough, release procedure, migration history
tests/       root guards: lockstep versions, sibling pins, workspace wiring
scripts/     bump_version.py — one version, one commit, one tag
.agents/     the gated agentic study protocol (bayescycle-study skill)
```

Workspace-level working discipline lives in [`AGENTS.md`](AGENTS.md); each
package has its own `AGENTS.md` with its identity and invariants. Releases
are lockstep: a `vX.Y.Z` tag publishes all five distributions
([`docs/releasing.md`](docs/releasing.md)).

## Development

```bash
uv run pytest tests -q                  # root workspace guards
uv sync --package <member> && cd packages/<member>  # then that package's
#   ruff format --check . && ruff check . && ty check && pytest
cd packages/bayesite-viz && uv sync     # viz packages are standalone
```

CI runs all of the above per PR, including the no-JAX profile with a real
pinned engine binary.
