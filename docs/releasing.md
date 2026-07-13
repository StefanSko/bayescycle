# Releasing

This repository is a **uv workspace monorepo** publishing five lockstep
Python distributions to PyPI: `bayeswire`, `bayesjax`, `bayescycle`,
`bayesite-viz`, `bayesite-idata` (see
[`monorepo-migration.md`](monorepo-migration.md) for how this replaced the
old four-repo layout). A release tags one `main` commit and publishes all
five at the same version number in one workflow run.

The wire format is still the product whenever a release carries a
canonical-bytes change: `bayeswire_ir` is the serialization boundary every
package, and the separate `bayesite` (Rust engine) repository, agrees on. A
corpus diff is a compatibility event with its own review, independent of the
version-bump mechanics below.

## When to release

- **Any change to canonical bytes, tags, or field lists** (a corpus diff
  under `packages/bayeswire/src/bayeswire/corpus/`) requires:
  1. A spec changelog entry and an explicit `bayeswire_ir` version decision
     in `spec/ir-format-v1.md`.
  2. A deliberate corpus regeneration, reviewed byte-by-byte:
     `uv run --package bayeswire packages/bayeswire/scripts/regenerate_corpus.py`.
  3. Regenerated evaluation fixtures from the JAX oracle:
     `uv run --package bayesjax packages/bayesjax/scripts/generate_ir_fixtures.py
     --bayeswire-path packages/bayeswire`.
  4. If the artifact corpus changed: regenerate with a real `bayesite`
     binary via `packages/bayeswire/scripts/generate_artifact_corpus.py
     --bayesite-bin ...`.
- **Package-only changes** (docs, tooling, additive corpus growth with
  byte-identical existing files) can be released whenever convenient;
  `bayeswire_ir` stays at 1.

## Preconditions

1. `main` CI green (`.github/workflows/ci.yml`): the root guard job, and the
   per-package `bayeswire` / `bayesjax` / `bayescycle` / `bayescycle-no-jax`
   / `viz-packages` jobs. Because bayeswire, bayesjax, and bayescycle share
   one workspace lock, HEAD-vs-HEAD compatibility across those three is
   exercised by every PR — there is no separate nightly cross-repo alignment
   run to wait on before tagging.
2. If the corpus changed: the diff was reviewed byte-by-byte, and the
   evaluation/artifact fixtures were regenerated as described above.
3. `CHANGELOG.md` has a complete dated section for the intended version.
   Every user-visible change since the previous tag is represented; internal
   implementation notes are not a substitute.

## Cut the release

One version, one commit, one tag, one workflow run:

1. **Finalize the changelog.** Move the accumulated `Unreleased` entries into
   `## [X.Y.Z] - YYYY-MM-DD`, restore an empty `Unreleased` section, and update
   the comparison links at the bottom of `CHANGELOG.md`.
2. **Bump.** `uv run python scripts/bump_version.py --version X.Y.Z` rewrites
   all five `pyproject.toml` versions; the exact sibling pins between them
   (bayescycle -> bayeswire, bayescycle's `[inproc]` extra -> bayesjax,
   bayesjax -> bayeswire); the runtime `__version__` constants; and the
   `BAYESITE_VIZ_SOURCE` / `BAYESITE_IDATA_SOURCE` / `BAYESITE_VIZ_EXCLUDE_NEWER`
   pins in
   `packages/bayescycle/src/bayescycle/backends/bayesite_viz/uvx_runner.py`.
   Refresh all lock files with `uv lock`, `cd packages/bayesite-viz && uv lock`,
   and `cd packages/bayesite-idata && uv lock`.
3. **Commit.** One commit. `uv run pytest tests -q` (the root guard suite)
   asserts all five versions, sibling pins, and the changelog section agree —
   run it before tagging.
4. **Tag.** `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin main vX.Y.Z`.
5. **Publish.** The tag push triggers `.github/workflows/release.yml`:
   - `check` re-verifies the tag matches the version read from
     `packages/bayeswire/pyproject.toml` and re-runs the root guard tests.
   - Five `publish` matrix jobs (one per package, each its own
     `pypi-<name>` trusted-publishing environment) build and
     `uv publish --check-url ...` a wheel/sdist. `--check-url` makes a re-run
     idempotent: artifacts already on PyPI are skipped, not re-uploaded.
6. **Release notes.** Create or update the GitHub Release for the tag from the
   matching `CHANGELOG.md` section.
7. **Verify.** `uv tool install bayescycle` from real PyPI and run the
   quickstart.
8. **Stage and smoke-test the Playground.** Run
   `uv run python playground/scripts/stage_assets.py`. This rewrites
   `playground/site/VERSION.json` from the lockstep package version, stages the
   sibling bayeswire source used by Pyodide, verifies the committed
   `bayesite_core.wasm` against `ENGINE.json`, and fetches the sha-pinned
   Pyodide distribution. Run `uv run pytest playground/tests -q`; the Pages
   workflow performs the same staging before deployment.
9. **Advance Bayesite's G7 oracle pin.** In the separate `bayesite` repository,
   update the exact `bayesjax==X.Y.Z` dependency in
   `scripts/check_rust_backend_posterior.py`, run G7, and merge the pin bump
   only when cross-backend conformance passes. If the release changed the wire
   corpus, combine this with the Bayeswire vendor refresh described below.

## The three surviving cross-repo edges

`bayesite` (the Rust engine) stays a separate repository; it vendors the
spec and fixtures by file, never by package dependency. Three pins move
independently of the lockstep Python release above:

1. **Wire change -> bayesite vendor refresh.** After any canonical-bytes
   change, run `scripts/vendor_bayeswire.py` in the `bayesite` repo, pointed
   at this monorepo (`packages/bayeswire` + root `spec/`). The vendored diff
   is generated bytes, never hand-edited; review it byte-by-byte the same
   way the corpus diff here is reviewed. A wire change is not done until the
   vendor refresh lands in `bayesite`.
2. **A bayesite release -> native and Playground engine pin bumps.** When
   `bayesite` ships a new tag, run
   `uv run python packages/bayescycle/scripts/bump_engine_release.py --tag
   vX.Y.Z` to rewrite `PINNED_ENGINE_RELEASE` (the version and per-target
   sha256s) in
   `packages/bayescycle/src/bayescycle/backends/bayesite/provisioning.py`.
   Separately replace `playground/site/vendor/bayesite/bayesite_core.wasm`
   with the release's wasm artifact and update `ENGINE.json` with its engine
   version, source commit, and sha256. Run the Playground corpus, engine, and
   browser-runtime tests before merging the wasm edge. These two consumers
   may move independently, but neither pin is implicit.
3. **A bayescycle release -> Bayesite G7 pin bump.** After the PyPI artifacts
   exist, update Bayesite's exact Bayesjax oracle dependency to the released
   lockstep version and run G7. Do not resolve an unpinned "latest" version:
   incompatibility must produce a deliberate Bayesite compatibility change,
   not a silently broken scheduled job.

These edges are not performed by `bump_version.py` or `release.yml`; each is a
reviewed follow-up in the repository that consumes the new release.

## Playground Pyodide edge

Pyodide is fetched only by the staging script and is pinned by version, URL,
and archive sha256 in `playground/site/vendor/pyodide/VENDOR.json`. To upgrade
Pyodide, update all three fields together, run the staging script from an empty
`playground/site/vendor/pyodide/` cache, and rerun the complete Playground
browser suite. Never update a mutable URL without updating and reviewing its
hash.

## Pending

A scheduled `bayesite HEAD vs workspace HEAD` job (the engine built from
source, run against this workspace's `main`) has not been re-created since
the migration — see [`monorepo-migration.md`](monorepo-migration.md) for
status. Until it exists, `bayesite` compatibility is checked at vendor-refresh,
engine-pin-bump, and released-Bayesjax G7-pin time, not continuously against
workspace `main`.
