# Monorepo migration plan

Decided direction (interview, 2026-07-07): merge the Python packages —
bayeswire, jaxstanv5, bayescycle, bayesite-viz, bayesite-idata — into one
uv-workspace monorepo publishing lockstep versions to PyPI. **bayesite
(Rust) stays a separate repo**: it keeps vendoring spec + fixtures by
byte-reviewed file copy and keeps shipping the engine as a released binary.

Motivation: the ordered four-consumer pin-bump dance in
[releasing.md](releasing.md) exists to preserve an independence (consumers
lagging on old bayeswire) that policy forbids and practice only used for
stability-during-experiments — which published lockstep versions serve
better (`bayescycle==<old>` is always installable).

Hard constraints that survive the migration:

- bayeswire the *package* keeps its identity: stdlib-only, enforced by the
  no-JAX module walk.
- bayesite-viz/bayesite-idata stay behind the uvx process boundary
  (`uvx_runner.py`) — never a path or wheel dependency of bayescycle. The
  arviz/matplotlib/netcdf4 stack must not enter bayescycle's closure;
  `uv tool install bayescycle` installs bayeswire and nothing else heavy.
- The engine reaches bayescycle as a pinned release binary
  (`PINNED_ENGINE_RELEASE`); joint engine+workflow dev uses the explicit
  engine-path option against a locally built binary.

Each step is red→green TDD from the current agreed state and ends with its
own docs updated — no step leaves a doc describing the previous world.
Process decision (2026-07-07): this migration runs on direct commits to
main, no PRs; implementation is delegated to lower-powered subagents
where mechanical.

## Step 0 — make the uvx invocation deterministic (in bayescycle, now)

Independent of the migration; closes a live determinism gap. Today
`uvx --from bayesite-viz@<commit>` resolves transitive `>=` deps at plot
time, so an upstream arviz release can break `bayescycle plot` with no
change on our side — the one nondeterministic resolution in the toolchain.

- **Red**: extend `tests/test_bayesite_viz_uvx_runner.py` — both
  `idata_command` and `plot_command` argv must carry
  `--exclude-newer=<BAYESITE_VIZ_EXCLUDE_NEWER>`, a constant timestamp
  pinned next to `BAYESITE_VIZ_SOURCE`.
- **Green**: add the constant (set to the pinned commit's date) and wire it
  into both argv builders in `uvx_runner.py`.
- Add a cache-warming preflight so offline runs fail early, not
  mid-workflow: a `bayescycle`-level command or documented
  `uvx --from ... --help` warmup; red test on the command construction
  first.
- **Docs**: releasing.md consumer-pin checklist gains "bump
  `BAYESITE_VIZ_EXCLUDE_NEWER` alongside `BAYESITE_VIZ_SOURCE`"; bayescycle
  README notes the determinism guarantee.

## Step 1 — monorepo scaffold with history

**Decision point (needs sign-off before executing): the monorepo home.**
Recommended: transform this repo (bayeswire) into the monorepo and rename
it (GitHub redirects old URLs, so existing git-tag pins keep resolving);
the alternative is a fresh sixth repo. Recommendation rationale: this repo
already owns the spec, the corpus, and the toolchain docs, and its history
moves by `git mv` instead of import.

- Layout: `packages/{bayeswire,jaxstanv5,bayescycle,bayesite-viz,bayesite-idata}`,
  root `pyproject.toml` declaring the uv workspace with
  `{ workspace = true }` sources; `spec/` and `docs/` stay at the root
  (the spec is toolchain-normative, not package-internal).
- Import jaxstanv5, bayescycle, bayesite-viz histories with
  `git filter-repo --to-subdirectory-filter` merges; bayesite-viz's own
  two-package workspace flattens into the root workspace.
- Replace all cross-package git-URL pins with workspace path deps;
  `[project.dependencies]` carry exact sibling versions (see step 3),
  `[tool.uv.sources]` overrides them in dev.
- **Red→green**: the merged test suites are the green bar; new red tests
  only where behavior is new — a workspace-level guard that bayescycle's
  built metadata depends on bayeswire alone, and that no package other
  than jaxstanv5 (and bayescycle's `[inproc]` extra) can import JAX.
- `projects/` sibling checkouts shrink to `projects/bayesite` only.
- **Docs**: root AGENTS.md/CLAUDE.md rewritten for the workspace (identity
  per package, one validation command block); per-package AGENTS.md where
  the old repo had one.

## Step 2 — workspace CI replaces cross-repo CI

- Per-package jobs: ruff / ty / pytest via `uv sync --package <name>`.
  The no-JAX install profile must sync **only** bayescycle+bayeswire (the
  workspace lock contains JAX; the profile proves the subset works without
  it, with a real provisioned engine binary) — port bayescycle's two
  existing install-profile jobs unchanged.
- The nightly cross-repo alignment dissolves: HEAD-vs-HEAD across Python
  packages is now every PR's CI. What survives as a scheduled job is
  `bayesite HEAD vs workspace HEAD` (the engine is still a separate repo).
- produce-conformance, the module walk, and spec-doc generation run as
  today, scoped to `packages/bayeswire`.
- **Red→green**: CI is config, not code — the red bar is a deliberately
  broken fixture per job (e.g. a JAX import in the no-JAX profile) proving
  each gate actually gates, then removed.
- **Docs**: toolchain.html CI description; delete the alignment-workflow
  docs.

## Step 3 — lockstep versioning and PyPI publishing

- **Precondition**: confirm PyPI name availability for all five
  distributions (`bayeswire`, `jaxstanv5`, `bayescycle`, `bayesite-viz`,
  `bayesite-idata`); reserve immediately. A taken name is a naming
  decision, not a blocker — flag it.
- One version for all five packages, starting at **0.3.0** (above every
  current per-repo version). A `scripts/bump_version.py` rewrites the five
  `pyproject.toml` versions and the exact sibling pins in
  `[project.dependencies]` in one commit; red test first: a repo-level
  test asserting all five versions and all sibling pins agree.
- `BAYESITE_VIZ_SOURCE` becomes `bayesite-viz==<bayescycle's own version>`
  derived from package metadata at runtime (red test:
  `_package_source`/source construction equals the running version);
  `BAYESITE_VIZ_EXCLUDE_NEWER` becomes the release timestamp, stamped by
  the release workflow.
- Publishing: PyPI trusted publishing (OIDC) workflow on tag `vX.Y.Z` —
  build all five wheels/sdists, publish atomically; dry-run against
  TestPyPI first with a smoke job that `uv tool install bayescycle
  --index testpypi` and runs a no-JAX workflow against a provisioned
  engine.
- **Docs**: releasing.md rewritten (see step 5 for the shape).

## Step 4 — first lockstep release and consumer cutover

1. Tag `v0.3.0` → five wheels publish; verify
   `uv tool install bayescycle` from real PyPI, run the quickstart.
2. bayesite: repoint `scripts/vendor_bayeswire.py` at
   `<monorepo>/packages/bayeswire`; the vendored byte diff must be **empty**
   (no wire change is allowed to ride the migration — if the corpus moved,
   stop and split).
3. Archive the three merged repos (README pointer to the monorepo; repos
   stay readable so historical git-URL pins keep resolving).
4. Grep the world for leftover git-URL pins to the old repos.

## Step 5 — docs sweep and retirement

The closing pass; nothing merges after this that describes the old world.

- **releasing.md**: replaces the four-consumer ordered checklist with the
  new shape — one monorepo tag publishes five packages lockstep; a wire
  change additionally requires the bayesite vendor PR (byte-diff review
  unchanged) and, when the engine releases, `bump_engine_release.py`.
  The two surviving cross-repo edges (vendor, engine pin) are the whole
  ceremony.
- **AGENTS.md / CLAUDE.md**: consumer list, `projects/` section, identity
  statements ("known consumers pin by exact version" → lockstep wheels).
- **toolchain.html**, per-package READMEs, `docs/invariants.md` (repo-layout
  references), bayesite's contributor docs (vendor source path).
- Delete: cut-tag ordering rules, merge-commit/SHA-reachability rules, the
  uv URL-conflict warning — each replaced by a one-line pointer to this
  plan's rationale where context helps.
- Grep for the old repo URLs and `v0.2.0`-style pin references across all
  docs before declaring done.

## Sequencing and stop rules

Steps are strictly ordered; each lands as its own commit series on main
(step 1 may be several).
Stop-and-consult points: the step 1 home/rename decision, any PyPI name
collision in step 3, and any non-empty vendored byte diff in step 4.
No wire-format (`bayeswire_ir`) change may ride any migration PR.
