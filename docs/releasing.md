# Releasing

The six Python distributions are versioned and published together from one
`main` commit and one `vX.Y.Z` tag. `bayesite` is released independently from
its Rust repository.

## Preconditions

- `main` CI is green, including root guards, all package jobs, the no-JAX
  Bayescycle profile, and the Playground browser suite.
- `CHANGELOG.md` has a complete dated section for the release.
- Any canonical-byte, tag, or field-list change has a spec changelog entry, an
  explicit `bayeswire_ir` version decision, and a byte-reviewed corpus diff.
- Changed corpus evaluation or artifact fixtures were regenerated with the
  Bayesjax oracle or a real Bayesite binary as appropriate.

## Cut the release

1. Move `Unreleased` changelog entries into `## [X.Y.Z] - YYYY-MM-DD`, restore
   an empty `Unreleased` section, and update comparison links.
2. Run:

   ```bash
   uv run python scripts/bump_version.py --version X.Y.Z
   uv lock
   (cd packages/bayesite-viz && uv lock)
   (cd packages/bayesite-idata && uv lock)
   ```

   The script updates all six versions, exact sibling dependencies, runtime
   version constants, and Bayescycle's `uvx` package pins.
3. Run `uv run pytest tests -q`, commit the release as one commit, and tag it:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main vX.Y.Z
   ```

4. `.github/workflows/release.yml` verifies the tag and root guards, then builds
   and publishes all six distributions through their PyPI trusted-publishing
   environments. Re-runs skip artifacts already present on PyPI.
5. Create the GitHub Release from the matching changelog section.
6. Verify a clean `uv tool install bayescycle` from PyPI and run a quick sample.
7. Stage and test the Playground:

   ```bash
   uv run python playground/scripts/stage_assets.py
   uv run pytest playground/tests -q
   ```

## Wire-format changes

Regenerate model documents, hashes, data documents, fingerprints, and the tag
spec deliberately:

```bash
uv run --package bayeswire packages/bayeswire/scripts/regenerate_corpus.py
uv run --package bayesjax packages/bayesjax/scripts/generate_ir_fixtures.py \
  --bayeswire-path packages/bayeswire
```

Review every changed byte. Existing corpus files must remain byte-identical
unless the version decision explicitly permits a breaking format change. When
artifact fixtures change, regenerate them with
`packages/bayeswire/scripts/generate_artifact_corpus.py --bayesite-bin ...`.

After the Python change is final, run `scripts/vendor_bayeswire.py` in the
Bayesite repository against this repository's `packages/bayeswire` and `spec/`.
The vendored diff is generated and byte-reviewed, never hand-edited.

## Independent Bayesite pins

A Bayesite release has two consumers here:

- Update native provisioning with
  `uv run python packages/bayescycle/scripts/bump_engine_release.py --tag vX.Y.Z`.
- Build `bayesite_core.wasm` from the exact release commit, replace the staged
  Playground module, and update `ENGINE.json` version, commit, and SHA-256.

Run the affected package, corpus, and browser tests before merging either pin.
The native and Wasm consumers may advance independently, but neither is
implicit.

After a Bayescycle release reaches PyPI, update Bayesite's exact Bayesjax G7
oracle dependency to the released version and run the cross-backend check.

Pyodide is separately pinned by version, URL, and archive SHA-256 in
`playground/site/vendor/pyodide/VENDOR.json`. Change all three together, stage
from an empty cache, and rerun the full browser suite.
