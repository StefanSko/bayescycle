# AGENTS.md

## What this repository is

`bayescycle` is a **uv workspace monorepo** publishing five lockstep Python
distributions to PyPI, plus a separate Rust engine repository it depends on
at a distance:

- **bayeswire** (`packages/bayeswire`) — the model declaration eDSL,
  resolved-metadata IR, the `bayeswire_ir` wire codec, dimension sidecars,
  the normative spec, and the golden conformance corpus. Stdlib only.
- **bayesjax** (`packages/bayesjax`) — the JAX/BlackJAX reference sampling
  backend: binds bayeswire models, compiles log densities, runs NUTS, and is
  the float64 oracle behind the corpus's evaluation fixtures.
- **bayescycle** (`packages/bayescycle`) — the Python workflow CLI: loads a
  model file, serializes IR, prepares a run directory, and invokes a
  backend. Default backend is `bayesite`; `bayesjax` is opt-in via the
  `[inproc]` extra.
- **bayesite-viz** (`packages/bayesite-viz`) — ArviZ plotting CLI, reached by
  `bayescycle` through `uvx` at a pinned version, never installed directly.
- **bayesite-idata** (`packages/bayesite-idata`) — run-directory-to-NetCDF
  exporter, reached the same way.
- **bayesite** — a **separate Rust engine repository**, not in this
  workspace. It vendors `spec/` and the bayeswire corpus by byte-reviewed
  file copy (never a package dependency) and ships released binaries that
  `bayescycle` provisions via `PINNED_ENGINE_RELEASE`.

Each package's own identity, invariants, and working discipline live in its
own `AGENTS.md` — read that before touching its code. This file is the
workspace-level view; it does not restate package content.

## Workspace layout

Root `pyproject.toml` declares a uv workspace with three members:

```text
packages/bayeswire, packages/bayesjax, packages/bayescycle
```

`packages/bayesite-viz` and `packages/bayesite-idata` live in the repo,
version lockstep with the workspace members, but are **standalone uv
projects, not workspace members** — each has its own `uv.lock`. This is
deliberate: the workspace lock resolves one set of packages together, so
adding these two would force their `arviz`/`matplotlib`/`netcdf4` stack to
co-resolve with the rest of the workspace, including `bayesjax`'s JAX stack.
`bayescycle` must stay installable with only `bayeswire` in its closure
(`uv tool install bayescycle` installs nothing heavy); keeping the viz
packages out of the workspace, reachable only through the `uvx` process
boundary, is what makes that true in practice, not just in principle.

`spec/` and `docs/` live at the repo root — the wire spec is
toolchain-normative, not internal to any one package.

## Validation

- **Root guards** — workspace wiring invariants (lockstep versions, sibling
  pin agreement, workspace-member boundaries): `uv run pytest tests -q`.
- **Each workspace member** — `uv sync --package <bayeswire|bayesjax|bayescycle>`,
  then that package's own `ruff format --check` / `ruff check` / `ty check`
  / `pytest` from inside `packages/<name>` (see its `AGENTS.md`).
  `bayescycle` additionally has a no-JAX install profile
  (`uv sync --package bayescycle`, without `[inproc]`) that must run without
  JAX ever entering the environment.
- **Viz packages** — standalone projects: `cd packages/bayesite-viz` (or
  `bayesite-idata`) and run `uv sync` / `ruff format --check .` /
  `ruff check .` / `ty check` / `pytest tests -q` from that directory; they
  are not reachable via `uv sync --package`. Each package carries its own
  `[tool.ty]` config so type checking resolves against its own `.venv`
  rather than the workspace root's.
- `.github/workflows/ci.yml` runs all of the above per-PR; there is no
  cross-repo nightly anymore for the three workspace members, since they
  share one lock and CI already exercises HEAD-vs-HEAD.

## Changelog and releasing

- `CHANGELOG.md` is the curated user-facing history for all five lockstep
  distributions. Every user-visible PR updates its `Unreleased` section, or
  explains in the PR why no entry is needed.
- Release preparation moves `Unreleased` entries into one dated version
  section. Root guards require a section for the current lockstep package
  version. `CLAUDE.md` is a symlink to this file, so both agent entry points
  carry the same rule.

One version, one commit, one tag: `scripts/bump_version.py` rewrites all
five package versions and their sibling pins, the root guard tests confirm
agreement, a `vX.Y.Z` tag triggers `.github/workflows/release.yml`, which
publishes all five distributions via per-package PyPI trusted-publishing
environments. `bayesite` compatibility (vendor refresh, engine pin) moves on
its own schedule, driven by events in that repository. Full procedure,
including the wire-format-change path, is in
[`docs/releasing.md`](docs/releasing.md).

## Further reading

- [`docs/releasing.md`](docs/releasing.md) — the release procedure.
- `packages/bayeswire/AGENTS.md`, `packages/bayesjax/AGENTS.md`,
  `packages/bayescycle/AGENTS.md` — per-package working discipline.
