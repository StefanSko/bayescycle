# The Playground v0 — implementation plan (pi-first)

**2026-07-12 · builds on:** bayesledger v1.1 (the source of the carve — walks
1–3, D0–D16) · the complexity review of 2026-07-12 (findings F1–F11, moves
M1–M5) · the Stan Playground investigation below. **No engine changes
required** — every verb v0 needs shipped in bayesite 0.2.1.

**Positioning (the publication ladder):** this is **ring 0** — the public
wedge. Model in, plots out, ten seconds to first sample, fully client-side.
The workflow discipline (phases, gates, seals — bayesledger's substance) is
the unstated northstar: ring 1 adds a saveable session file, ring 2 makes it
append-only with provenance, ring 3 is the Ledger, arriving for users who
already live in the tool. Nothing from bayesledger v1.1 is wasted; it becomes
the working prototype of ring 3 and the parts bin for rings 0–2. Lead with
the toy, accrete the philosophy.

**Division of labor (pi-first):** Claude owns design, spec-freezing, and diff
review. Pi (one session per work order) owns implementation from frozen
specs. Commits/pushes/deploys stay Claude-side. Every work order ships with
observable success criteria and exact validation commands; Pi's claims are
advisory until Claude re-runs them.

## Prior art — Stan Playground (investigated 2026-07-12)

Facts, verified against the repo and docs
(github.com/flatironinstitute/stan-playground):

- **Architecture:** React + Vite frontend; **compilation requires a
  dedicated server** (FastAPI + Docker; Flatiron hosts a public default;
  self-hosting needs Docker). Each Stan model is compiled server-side via
  tinystan/emscripten into a **per-model wasm artifact** the browser
  downloads. Sampling then runs locally — real Stan NUTS/autodiff in wasm.
  Their own docs: *"Compilation of the models is the only part of Stan
  Playground which is not run locally."*
- **Project files:** `main.stan`, `data.json`, `data.py`/`data.R`
  (data-generation scripts run **in the browser** via Pyodide/webR — their
  `transformed data` analog), `analysis.py`/`analysis.R` (post-sampling
  scripts, same runtimes), plus an "additional files" tab (upload a CSV,
  scripts can `read.csv()` it).
- **Sampling options:** `num_chains`, `num_warmup`, `num_samples`,
  `init_radius`, `seed` — all URL-parameterizable.
- **Outputs:** summary-statistics table, trace plots, histograms, scatter
  plots; anything further is DIY in analysis scripts.
- **Sharing:** download/upload project; GitHub Gist integration; quick-share
  with the whole project in the URL (size-limited); URL params that load
  individual files from anywhere on the web; an **embed mode** for docs and
  course pages.
- **Positioning:** teaching and experimentation. Wasm memory ceiling
  (2–4 GB per tab) applies to them as to us.

### The differentiator (say it everywhere)

Stan's path is `stanc → C++ → emscripten` — a compiler pipeline that cannot
run in a tab, hence the server and the per-model artifact. Our path is
`model.py → Pyodide → IR → bayesite.wasm` — **interpretation, not
compilation**. One committed ~1 MB engine wasm executes every model; there
is no server, no Docker, no per-model download, and the same IR hash and
artifact bytes are reproducible in the native CLI. Honest claims for the
landing blurb: *"no server anywhere — your model and data never leave the
tab"* and *"paste the IR hash into the CLI and get byte-identical results."*
(Full offline-after-first-load needs the service worker — that is ring 0.5,
don't claim it early.)

### Steal list (they got these right)

Additional-files tab · embed mode for teaching · URL params that load remote
files · `analysis.py` post-run scripting (cheap for us — Pyodide is already
resident; ring 0.5) · init-radius-style sampler control surfaced plainly.

## Confirmed decisions

| Decision | Choice |
|---|---|
| Location | **this monorepo**, top-level `playground/` — in-repo, lockstep-versioned, **not** a uv workspace member, **not** an npm project (precedent: bayesite-viz/-idata "in the repo, own toolchain") |
| Stack | **buildless**: native ES modules, plain JS + JSDoc discipline (no TypeScript toolchain, no bundler, no `package.json` anywhere), vendored **prebuilt** CodeMirror 6 bundle carried over from bayesledger |
| bayeswire | staged from sibling `packages/bayeswire/src/bayeswire` by a checked-in script — an **intra-repo reference**, not a vendored copy; corpus fixtures likewise. Former edge E4 ceases to exist |
| Engine | committed `playground/vendor/bayesite/bayesite_core.wasm` + `ENGINE.json` (version, commit, sha256) — the one surviving cross-repo edge, **registered in `docs/releasing.md` as part of this plan** (WO-P6), not left invisible again |
| Pyodide | fetched at CI/deploy time; version + sha256 pinned in a **committed** `playground/vendor/pyodide/VENDOR.json` sidecar (fixing bayesledger's gap F11) |
| sqlite-wasm | **dropped** — no study file, no SQL scratch step; CSV → typed columns in plain JS, with a one-click standardize toggle (the F7 lesson: nobody should hand-write variance SQL) |
| Tests | unit: **`node --test`** on `.mjs` modules (Node ships the runner; zero npm); e2e: **pytest + playwright-python** inside the uv workspace — one CI toolchain for the whole repo |
| Deploy | GitHub Pages, static directory copy, versioned with the lockstep release; a version chip in the UI shows it |
| Carve source | bayesledger @ v1.1 by copy-and-adapt (not submodule): `src/compile`, `src/engine`, `src/dashboard`, `src/critique/render` (overlay + overlap coefficient), `src/ui/scratch-data` (delimiter sniffing), the sampler-settings pane (F-I) and design-value defaults (F-B) |
| Naming | directory `playground/`; public name decided at WO-P6 before deploy |

**Client-side invariant (inherited verbatim):** no backend, no telemetry, no
third-party runtime requests; the only network use is the app fetching its
own static assets. Model and data never cross the network. CI asserts it —
for real this time (WO-P0 includes the origin-isolation job bayesledger's
CONVENTIONS promised but never had).

**Non-goals for v0 (recorded, not silent):** no `.bayes` file or any
persistence beyond downloads · no phases/gates/seals/DAG/reading mode · no
service worker (ring 0.5) · no `analysis.py` scripting (ring 0.5) · no R ·
no accounts, ever.

## Milestones

- **MP1 — Headless core.** Compile + engine + data ops live in the monorepo
  and pass under `node --test`, byte-identical to the sibling corpus.
- **MP2 — The loop.** Model → data → sample → plots, streaming, in a tab.
- **MP3 — The honest playground.** Simulate + prior-predictive on design
  values (the "transformed data, but truth-known" moment), share links,
  examples menu.
- **MP4 — Published.** Dogfood walk clean, deployed to Pages, edge
  registered, announcement drafted. v0 done.

## Work orders

### WO-P0 · Scaffold — *Pi, small*
`playground/{src,vendor,tests,scripts,public}`; `scripts/stage_assets.py`
(uv-run: copies sibling bayeswire source + corpus fixtures into
`playground/public/vendor/`, fetches Pyodide against the committed sha,
verifies the engine wasm sha); CI additions to `ci.yml`: `node --test`
job, `uv run pytest playground/tests -q` job, and the origin-isolation job
(serve the static dir, block third-party hosts, assert boot + compile).
**Validate:** all three CI jobs green on the skeleton; `git grep -l
package.json playground/` returns nothing.

### WO-P1 · Core port — *Pi, core, blocks everything; thinking xhigh*
Port `compile/` (Pyodide runtime + bayeswire mount) and `engine/` (ABI glue,
per-chain workers, `per_draw_v2` stream parse, verb functions minus
studyfile writes — results return in memory) from bayesledger v1.1 to
buildless ESM + JSDoc. **Spec:** bayesledger's `spec/verbs-v0.md` §2–§3
carries over minus every "appends event" clause; results are plain objects.
**Validate:** `node --test` — all 10 staged corpus models: Pyodide-computed
IR hash **byte-identical** to `corpus/hashes.json`; golden `diagnose`
byte-identical on the staged fixture streams; malformed IR surfaces the
typed engine error.

### WO-P2 · Data module — *Pi, parallel with WO-P1 after WO-P0*
CSV upload (delimiter sniffing + schema preview + wrong-delimiter warning,
ported from WO-14), JSON paste, typed column store in plain JS, per-column
standardize toggle, and the binding layer that matches columns to the
model's declared `Data` inputs by name with a visible mapping table.
**Validate:** `node --test` on sniffing/typing/standardize (mean 0, sd 1
within float tolerance); WaffleDivorce.csv (semicolon) yields 13 columns.

### WO-P3 · UI shell — *Pi, 2 orders; closes MP2*
Three-pane layout: model (CodeMirror, live debounced compile, IR-hash chip,
verbatim bayeswire errors) · data (WO-P2 surfaces) · run+results (settings:
chains/warmup/draws/seed/target_accept/max_treedepth; per-chain streaming
progress with divergence counts; auto-diagnose on completion). Results:
trank grid (default) / traces (toggle), ESS×R-hat scatter with verdict
line, precis dot chart, ppc density overlay, prior→posterior overlay with
the overlap-coefficient label (WO-15 port). Downloads: fit NDJSON,
diagnostics JSON, every plot as SVG.
**Validate:** playwright-python e2e — corpus model with inline data:
author → compile (hash chip) → bind → sample (progress streams) → all five
plot kinds render → downloads produce byte-stable SVGs on a fixed seed;
zero third-party requests (route interception); axe pass.

### WO-P4 · Simulate + prior-predictive — *Pi, after WO-P1; closes MP3 with WO-P5*
Design-values form for declared data inputs (constraint-aware defaults —
the F-B port: Positive → 1, never 0), truth form for parameters;
`prior-predictive` and `simulate` run with **zero observed data**; one-click
"sample on simulated data" chains simulate → sample and overlays truth
markers on the precis chart. This is the honest upgrade over `data.py`:
the generative model itself is the data generator, and recovery is visible.
**Validate:** e2e — divorce generative model: simulate at chosen truth →
sample → precis shows truth markers inside intervals on a fixed seed;
prior-predictive renders with no data bound at all.

### WO-P5 · Share links + examples — *Pi, small*
URL-fragment codec: project (model source, data-values/truth forms,
sampler settings — **not** uploaded data) → `CompressionStream('deflate-raw')`
→ base64url in `#`; decode path shows an interstitial — *"this link contains
model code; review before running"* — with the source visible and a Run
button (never auto-execute). Length guard warns near 8 kB. Examples menu:
the staged corpus models + the divorce pair, one click to load.
**Validate:** `node --test` round-trip property (project → fragment →
project, byte-equal); e2e — share a model, open in a fresh context,
interstitial shows, run reproduces the IR hash; oversized project warns.

### WO-P6 · Dogfood walk + publish — *human + Claude, no Pi*
Stefan + Claude walk the playground against the Stan Playground feature
table above — every row either matched, beaten, or consciously ring-0.5'd.
Findings → fix orders; loop until clean. Then: GitHub Pages workflow (on
lockstep tag), version chip wired, **`docs/releasing.md` gains the engine
edge** (wasm refresh = explicit step with its sha check) **and the staging
note** (playground assets regenerate from siblings at release), CHANGELOG
entry, public name decision, announcement draft naming the differentiator.
**v0's definition of done: the deployed URL samples eight-schools in a
fresh browser with the network tab showing zero non-origin requests.**

## Ring 0.5 backlog (recorded, not silent)

Service worker / true offline (port WO-9.5, incl. the F2 update-flow fix) ·
`analysis.py` on the resident Pyodide with draws in scope · Gist +
embed mode (the teaching channel) · additional-files tab ·
`recover-check` UI · session save/load — **ring 1's gateway, where the
`.bayes` file quietly returns.**

## Cadence per work order

Unchanged from the bayesledger plan: Claude freezes the spec first; Pi runs
via temp-file work order with recorded `--session-id`, thinking `high`
minimum, `xhigh` for WO-P1; Claude reviews the full diff and re-runs
validation personally; after 2 failed rounds Claude takes over; commit/push
Claude-side only. **Parallelism:** WO-P1 ∥ WO-P2 after WO-P0; WO-P4 ∥ WO-P5
after their deps; everything else sequential.

**Claude-only throughout:** spec authorship, the feature-table judgment in
WO-P6, naming, review, git mutations, and the two invariants no diff may
violate: **client-side only** (CI-enforced) and **zero npm** (there is no
package.json to audit — the invariant is the absence).

## Relation to bayesledger

The carve is by copy at v1.1, with a provenance line in each ported file
header. bayesledger itself proceeds per the decision record (demote to
pinned exhibition, or archive with honors) — that decision is independent
of this plan and this plan does not reopen it. The uvx CLI track
(`uvx bayescycle` + the bayescycle-study skill) remains the agent-facing
door; the playground is the human-facing one. Two front doors, one core,
one version.
