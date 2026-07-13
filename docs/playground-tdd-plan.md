# Playground v0 — red→green TDD execution plan (pi-first, all-browser)

**2026-07-12 · rev 2 · companion to** [`playground-plan.md`](playground-plan.md).
That document owns scope, decisions, and milestones; this one owns execution:
what was verified against reality, the amendments adopted, the all-browser
test architecture, and the red→green order of operations for every Pi work
order. Rev 2 supersedes rev 1's `node --test` design — **no Node toolchain
anywhere**; every test executes in a real browser.

## Verified against reality (2026-07-12)

| Claim in the plan | Status |
|---|---|
| Corpus: 10 models + `hashes.json` | ✓ `packages/bayeswire/src/bayeswire/corpus/` — exactly 10, with `data/` and `fixtures/` per model |
| Golden artifacts for diagnose | ✓ `corpus/artifacts/{eight_schools_non_centered,varying_intercepts_poisson}/` — `posterior.ndjson`, `diagnostics.json`, `manifest.json`, `model.ir.json`, `dims.json` |
| Engine wasm, verbs shipped | ✓ bayesledger `public/vendor/bayesite/`: `bayesite_core.wasm` + `ENGINE.json` (engine 0.2.1, bayesite commit `7164a05`, sha256 `fa7cc0f2…`) |
| Pyodide pin available | ✓ bayesledger `vendor.sh`: 314.0.2 core archive, sha256 committed |
| Frozen verb spec | ✓ bayesledger `spec/verbs-v0.md` §1–§4 frozen 2026-07-11 |
| CodeMirror prebuilt bundle | ✓ bayesledger `vendor/codemirror/codemirror.mjs` + `VENDOR.json` |
| CSV fixture | ✓ bayesledger `fixtures/divorce/WaffleDivorce.csv` (semicolon-delimited) |
| CI to extend | ✓ `ci.yml` per-package job structure |

## Amendments (adopted)

- **A1 · Carve pin = bayesledger commit `7346d71`.** No git tags exist there,
  and the F-I sampler-settings pane the plan ports landed one commit after
  the "v1.1 run complete" commit. Every ported file carries a provenance
  header naming this hash.
- **A2 · Translation, not copy.** Carve sources and their tests are
  TypeScript under vitest; the target is buildless ESM + JSDoc. Red-first
  fits naturally: port each module's tests first, then make them pass.
- **A5 · WO-P3 splits** into P3a (pure plot renderers) and P3b (UI shell).
- **A7 · "Zero npm" is executable**: a guard test asserts no `package.json`
  under `playground/` and that `playground/` is not a uv workspace member.
- *(Rev 1's A3/A4/A6 — Pyodide-under-Node, worker seam as test constraint,
  Node version pin — dissolved with the all-browser decision. The
  pure-core / worker-adapter split in the engine port survives as good
  design, no longer a test requirement.)*

## Test architecture — all-browser, one toolchain

**No Node anywhere.** The repo's single test toolchain is uv + pytest +
playwright-python (chromium). Two layers, both in a real browser:

1. **Unit suites** — plain `.mjs` test modules under
   `playground/tests/unit/*.test.mjs`, executed *inside the page* by a tiny
   committed harness (`playground/tests/harness.html` + `harness.mjs`,
   ~50 lines: dynamic-import the suite, run registered cases, expose
   `window.__results` as `[{name, ok, error}]`). One pytest test per suite
   file: serve `playground/public` statically, load the harness with
   `?suite=<name>`, await results, assert all green and report failures
   verbatim. Pyodide and the engine wasm run exactly where they ship — the
   browser — so byte-identity checks test the real environment, not a proxy.
2. **e2e flows** — pytest + playwright driving the actual app UI (the same
   serve fixture), including route-interception proof of zero non-origin
   requests and axe accessibility passes.

**One command validates everything:** `uv run pytest playground/tests -q`
(playwright-python added to the workspace dev group; CI job runs
`playwright install chromium` first).

**Rodney is the Claude-side instrument, not CI.** At every gate Claude
drives the served app with `rodney` (screenshots, `rodney js`, `ax-tree`,
click/input flows) to see what the tests claim. The WO-P6-style dogfood
loop is conducted with rodney until the UI/UX is judged good, findings
becoming fix orders.

## Cadence — red→green on pi-first

Per work order, **two Pi passes in one session** (same `--session-id`,
temp-file work order, `openai-codex/gpt-5.6-sol`, thinking `high`; `xhigh`
for WO-P1 green):

1. **RED (Pi).** Tests + fixtures + module **stubs only**. Stubs throw
   `Error("unimplemented")` so failures are assertion failures, never
   missing imports. Evidence: the pytest run showing N failing / 0 passing,
   pre-existing suites still green.
2. **Claude gate.** Read the red diff; every test must assert the frozen
   spec (byte-identity, error shapes), not stub behavior. **Tests freeze.**
3. **GREEN (Pi, same session).** Implement until green. Hard constraint:
   *test files are read-only; if a test looks wrong, stop and report.*
4. **Claude gate.** Re-run validation personally, review the full diff,
   probe the running app with rodney. Commit Claude-side.

After 2 failed green rounds, Claude takes over directly.

## Work orders

Dependency chain (sequential in one repo — no parallel Pi sessions):
**P0 → P1 → P2 → P3a → P3b → P4 → P5 → dogfood loop → PR.**

### WO-P0 · Scaffold
**RED:** guard pytest (A7 + staged-asset sha checks: engine wasm sha ==
`ENGINE.json`, Pyodide == `VENDOR.json`); harness smoke suite (boot Pyodide
in-page, run `1+1`; instantiate `bayesite_core.wasm`); e2e smoke (page
boots, zero non-origin requests).
**GREEN:** `playground/{src,public,tests,scripts}`;
`scripts/stage_assets.py` (stage sibling bayeswire + corpus fixtures into
`playground/public/vendor/`, fetch Pyodide against the committed sha, copy
engine wasm + `ENGINE.json` from the pinned bayesledger checkout with sha
verification, copy the CodeMirror bundle + `VENDOR.json`); the test harness;
playwright-python in the workspace dev group; one new `ci.yml` job
(`playground`: stage assets, install chromium, `uv run pytest
playground/tests -q`).
**Validate:** `uv run pytest playground/tests -q` · `uv run pytest tests -q`
· `git grep -l package.json -- playground/` empty · rodney: open the served
page, screenshot.

### WO-P1 · Core port — *green at thinking xhigh*
**RED** (translate bayesledger `compile/index.test.ts` +
`engine/index.test.ts` to harness suites, plus spec tests): for **all 10
corpus models**, in-browser Pyodide-computed IR hash byte-identical to
`corpus/hashes.json`; compile error surfaces the verbatim bayeswire error;
golden `diagnose` byte-identical on the staged
`eight_schools_non_centered` / `varying_intercepts_poisson` artifact
streams; `per_draw_v2` stream parse; malformed IR → typed engine error.
**GREEN:** translate `src/compile` (Pyodide runtime + bayeswire mount) and
`src/engine` (ABI glue, verbs, stream parse; pure core separate from the
worker adapter) minus every studyfile write — results are plain in-memory
objects; provenance headers with `7346d71`.
**Validate:** pytest green; Claude recomputes one model's IR hash via the
native workspace path and diffs bytes.

### WO-P2 · Data module
**RED:** translate `scratch-data.test.ts` (delimiter sniffing,
wrong-delimiter warning); typed column store; per-column standardize
(mean 0, sd 1 within float tolerance); staged `WaffleDivorce.csv`
(semicolon) yields 13 columns; binding layer matches columns to declared
`Data` inputs by name with a mapping-table model (unmatched inputs listed,
never silently dropped).
**GREEN:** CSV upload path, JSON paste, column store, standardize, binding.

### WO-P3a · Plot renderers
**RED:** harness suites against fixture draws for all six renderers — trank
grid, traces, ESS×R-hat scatter with verdict line, precis dot chart, ppc
density overlay, prior→posterior overlay with overlap-coefficient label.
Assert structural SVG properties + byte-stable snapshot on fixed input.
**GREEN:** port `src/dashboard/render` + `src/critique/render` as pure
`(draws, opts) → SVG string` functions; no DOM.

### WO-P3b · UI shell — *closes MP2*
**RED:** e2e specs against the stub shell: author → live debounced compile
(IR-hash chip) → bind → sample (per-chain streaming progress, divergence
counts) → auto-diagnose → five plot kinds render → downloads (fit NDJSON,
diagnostics JSON, SVGs byte-stable on fixed seed) → zero third-party
requests → axe pass.
**GREEN:** three-pane layout, vendored CodeMirror, sampler settings pane
(chains/warmup/draws/seed/target_accept/max_treedepth — F-I port), per-chain
Web Worker adapter, wiring to P1/P2/P3a.
**Validate:** pytest + a full rodney walk with screenshots.

### WO-P4 · Simulate + prior-predictive — *closes MP3 with P5*
**RED:** unit — constraint-aware design-value defaults (Positive → 1, never
0; the F-B port), truth form for parameters. e2e — divorce generative model:
simulate at chosen truth → one-click "sample on simulated data" → precis
shows truth markers inside intervals on a fixed seed; prior-predictive
renders with zero data bound.
**GREEN:** forms, zero-observed-data `simulate`/`prior-predictive`,
simulate→sample chaining, truth-marker overlay.

### WO-P5 · Share links + examples
**RED:** unit — round-trip property (project → `deflate-raw` → base64url
fragment → project, byte-equal, unicode included); length guard warns near
8 kB. e2e — shared link opens in a fresh context to the interstitial
(source visible, Run button, never auto-executes); running reproduces the
IR hash; oversized project warns.
**GREEN:** codec, interstitial, examples menu (corpus models + divorce pair).

### Dogfood loop — *Claude + rodney, no Pi*
Walk the full app with rodney against `playground-plan.md`'s Stan Playground
feature table: every row matched, beaten, or consciously ring-0.5'd.
UI/UX findings → fix orders (Pi or direct); loop until clean. This absorbs
WO-P6's walk; Pages deploy, releasing.md edge registration, naming, and
announcement remain post-merge follow-ups.

## Delivery

Feature branch `playground-v0` off `main`; one commit per work order
(Claude-side only). When the dogfood loop is clean: push, open PR, comment
`@codex review`, address findings, re-request until clean, then merge.

## Standing constraints (every Pi work order carries these verbatim)

- Do not commit, push, or touch anything outside `playground/`,
  `.github/workflows/ci.yml`, and (WO-P0 only) the root `pyproject.toml`
  dev group and root guard tests.
- No `package.json`, no npm, no bundler, no TypeScript syntax — plain `.mjs`
  + JSDoc.
- No network calls at runtime except same-origin static assets; staging
  scripts may fetch only the sha-pinned Pyodide archive.
- Green phase: test files are read-only.
