# bayescycle

**A file-based Bayesian workflow that agents and humans can replay and audit.**

```text
model.py -> bayeswire IR -> bayesite/bayesjax -> run/ -> NetCDF -> plots
```

Every phase reads and writes explicit artifacts. Runs are seeded, run
directories are append-only, and provenance records the inputs and backend that
produced each result.

## Toolchain

This repository publishes five lockstep Python distributions:

| Distribution | Responsibility |
|---|---|
| [`bayeswire`](packages/bayeswire) | Stdlib-only model eDSL, resolved IR, normative [`spec/`](spec/), and conformance corpus |
| [`bayescycle`](packages/bayescycle) | Workflow CLI, run and optional study-state directories, backend invocation, replay |
| [`bayesjax`](packages/bayesjax) | Optional JAX/BlackJAX NUTS backend and float64 oracle |
| [`bayesite-idata`](packages/bayesite-idata) | Run directory to ArviZ DataTree/NetCDF |
| [`bayesite-viz`](packages/bayesite-viz) | ArviZ plotting CLI |

[Bayesite](https://github.com/StefanSko/bayesite) is the external
zero-dependency Rust engine used by default. Bayescycle provisions a pinned,
SHA-256-verified release. The visualization distributions run through pinned
`uvx` environments and do not enter Bayescycle's default dependency closure.

## Quickstart

```bash
uv tool install bayescycle
bayescycle sample model.py --data data.json -o run/
bayescycle diagnose run/
bayescycle posterior-check run/ --seed 456
bayescycle plot trace run/
```

The first Bayesite-backed command provisions the pinned engine if necessary.
The optional in-process backend is installed explicitly:

```bash
uv tool install 'bayescycle[inproc]'
bayescycle sample model.py --data data.json -o run-jax/ --backend bayesjax
```

See [`packages/bayeswire`](packages/bayeswire) for model authoring and
[`packages/bayescycle`](packages/bayescycle) for generation, replay, engine,
study-state, and visualization commands. The browser Playground is documented in
[`playground/README.md`](playground/README.md).

## Design constraints

- `spec/` and the golden corpus are the interoperability contract.
- Decoding IR executes no user model code.
- Bayeswire is stdlib-only; the default Bayescycle path contains no JAX.
- Bayesjax and Bayesite must agree where conformance coverage says they do.
- Exact model and data bytes identify a fit; consumers never reserialize to
  compute hashes.

## Development

The root uv workspace contains Bayeswire, Bayesjax, and Bayescycle. The two
visualization projects are standalone uv projects with independent lock files.

```bash
uv run pytest tests -q
uv sync --package bayescycle
cd packages/bayescycle
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Working rules are in [`AGENTS.md`](AGENTS.md); package-specific invariants live
beside each package. Releases are described in
[`docs/releasing.md`](docs/releasing.md).
