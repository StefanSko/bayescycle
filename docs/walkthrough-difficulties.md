# Walkthrough regeneration — difficulties log

Notes captured while regenerating the workflow walkthroughs against the current
`main` (refactored integration modes + append-only `bayescycle.run.v1`
provenance). These are friction points to smooth out in a follow-up. None of
them are fixed here — the walkthroughs work around each one and point back to the
numbered item.

The runs were driven by `docs/build-walkthroughs.sh`, which produces two
non-linear demo trees (`complete/` single-backend Bayesite, `mixed/`
Bayesite→jaxstanv5 handoff), regenerates both HTML pages, and builds
`docs/walkthrough-runs.sqlite`.

## 1. Engine preflight can't read the current `bayesite --help` — RESOLVED

`bayescycle.backends.bayesite.preflight` discovers engine capabilities by running
`bayesite --help` and scanning for `usage: bayesite <cmd>` lines with
`^.*?usage:\s*bayesite\s+([A-Za-z0-9-]+)` under `re.MULTILINE`.

The current engine answers `--help` (and any unknown command) with a **single
JSON object on stderr**:

```json
{"error_format":"v0-provisional","error":"InvalidSettings","message":"unknown command \"--help\"; usage: bayesite sample ...\nusage: bayesite diagnose ..."}
```

The `\n` separators are JSON-escaped (literal backslash-n), so the whole probe
output is one physical line and the `^`-anchored regex matches only the **first**
command, `sample`. Every other stage (`simulate`, `recover`, `sbc`, `diagnose`,
`posterior-predictive`, `posterior-check`, `recover-check`) is then reported as
*"Selected Bayesite engine does not support required command … the binary may be
stale"* — even with a freshly built engine.

The bayescycle test fixture (`tests/test_cli.py::FAKE_BAYESITE_USAGE`) expects a
**multi-line plain-text** `--help`, which the real engine does not emit.

**Workaround:** `build-walkthroughs.sh` wraps the engine in a tiny shim that
answers the capability probe with parseable multi-line usage and `exec`s the real
binary for everything else.

**Resolved (bayeswire migration):** preflight now JSON-decodes single-line
engine errors and expands the embedded `message` text before scanning for
`usage:` lines, so the real binary's capability probe parses without a shim.
`build-walkthroughs.sh` drives the engine directly, and the real-engine
end-to-end test (`tests/test_bayesite_end_to_end.py`, exercised by the no-JAX
CI job) keeps it that way.

## 2. Cross-backend posterior-predictive / posterior-check is blocked by a fingerprint check

`bayesite posterior-predictive` and `bayesite posterior-check` recompute a
`model_data_fingerprint` from the supplied model + data and require the fit's
header fingerprint to match (`predictive.rs`: *"fit model_data_fingerprint must
match the supplied model and data"*).

The jaxstanv5 in-process backend writes its **own** `model_data_fingerprint`,
computed differently from Bayesite's. So a jaxstanv5-produced `posterior.ndjson`
is rejected by Bayesite's predictive checks. The previous walkthrough's flow —
`sample --backend jaxstanv5` then Bayesite `posterior-predictive` /
`posterior-check` — no longer runs.

**Workaround:** the complete walkthrough does the real fit on **Bayesite** so the
`diagnose → posterior-predictive → posterior-check` chain is fingerprint-clean.
The mixed walkthrough exercises the in-process fit but stops at
`recover-check`/`diagnose`, both of which read only the fit (no model+data
fingerprint recomputation) and so work cross-backend.

**Fix options:** a shared canonical fingerprint algorithm across backends, or a
neutral/`null` fingerprint that the predictive checks accept by falling back to
parameter-shape compatibility only.

## 3. `recover-check --targets` schema is unobvious

`targets.json` must be `{"targets": [{"name": "...", "truth": "...", "posterior":
"..."}]}`. A natural-looking flat map `{"alpha": 1.25, ...}` fails with
*"recover-check targets has unknown field \"alpha\""*. Omitting `--targets`
auto-maps each truth name to the matching posterior parameter, which is the
simplest path.

## 4. `recover` / `sbc` scenario schema needs version markers

Scenario files require a marker field or they're rejected with a terse message
(*"recover scenario needs recover_scenario \"v0-provisional\""*):

```json
{"recover_scenario": "v0-provisional", "data": {"x": [...]},
 "sample": {"chains": 4, "warmup": 400, "draws": 500, "target_accept": 0.9}, "seed": 7}
```

`sbc` is the same with `"sbc_scenario": "v0-provisional"` and an optional
`"replicates"`. The `sample` object accepts only
`chains|warmup|draws|max_treedepth|target_accept`. None of this is captured in
the run-directory docs.

## 5. `bayesite-viz` install is fragile in restricted environments

The README and old walkthrough install bayesite-viz with
`uvx --from git+https://github.com/StefanSko/bayesite-viz.git@<commit>`. In a
sandboxed environment that external-code execution is **denied**, and a pinned
commit drifts from the current viz `main`.

bayesite-viz also requires `python >= 3.13`, while bayescycle targets `3.12`, so
it can't share the project virtualenv.

**Workaround:** use a local checkout —
`uv run --no-project --with /path/to/bayesite-viz --python 3.13 -- bayesite-viz …`.
The driver takes the checkout path from `$BAYESITE_VIZ`.

## 6. `bayesite-idata` can't read the canonical `data.json`

`bayesite-idata` reads `run/data.json` expecting a **top-level** map
`{var: {shape, values}}`. bayescycle now writes the canonical wrapper there:

```json
{"format": "bayescycle.data.json.v1", "variables": {"x": {...}, "y": {...}}}
```

idata then fails with *"data.format has unsupported data value shape"* (it tries
to treat the string `"bayescycle.data.json.v1"` as a variable). The unwrapped
layout it wants only exists at `run/.bayesite/data.json`.

**Workaround:** the driver stages `model.ir.json`, `posterior.ndjson`,
`dims.json`, `posterior_predictive.ndjson` and the **unwrapped**
`.bayesite/data.json` into a temp directory and runs `bayesite-idata` against
that.

**Fix options:** teach bayesite-idata the `bayescycle.data.json.v1` wrapper, or
have it prefer `run/.bayesite/data.json` when present.

## 7. `uv run` must execute from the project directory

`uv run --extra inproc bayescycle …` warns *"--extra inproc has no effect when
used outside of a project"* and fails to find the `bayescycle` entry point when
the working directory is a demo/run directory (which is common, since model files
live there). The driver runs every `bayescycle` invocation inside a
`( cd "$PROJECT" && … )` subshell with absolute paths.

## 8. Generator was written for the previous (linear, n=1000, jaxstanv5-fit) flow

The old `workflow-walkthrough.py` assumed a single straight-line run whose real
fit used `--backend jaxstanv5`. The report **field** schemas (recovery_check,
recovery, sbc, posterior_check, diagnostics, posterior header/trailer) were still
compatible, so the rewrite was mostly narrative: the rejected first iteration, the
single-backend framing, and the new `run.json` provenance section. Worth a shared
fixture so the generator and the engine schemas can't silently drift again.

## 9. Importing the model leaves `__pycache__` in the demo tree

Loading `model.py` / `model_v1.py` writes `__pycache__/` into the demo
directories. The SQLite scanner only looks for `run.json`, so it's harmless, but
the trees aren't pristine. Minor; the driver could set
`PYTHONDONTWRITEBYTECODE=1`.
