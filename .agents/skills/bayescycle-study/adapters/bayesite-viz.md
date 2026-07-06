# Adapter: bayesite-viz

`bayescycle idata` and `bayescycle plot` are the visualization boundary for
bayescycle studies. They are first-class bayescycle CLI verbs that reach
`bayesite-viz` through a single pinned `uvx` source
(`BAYESITE_VIZ_SOURCE` in `src/bayescycle/backends/bayesite_viz/uvx_runner.py`
in the bayescycle repository), so there is nothing to install by hand.

Repository: <https://github.com/StefanSko/bayesite-viz>

`bayescycle` provides two agent-operable verbs over that boundary:

- `bayescycle idata`: bayescycle run directory -> ArviZ NetCDF/DataTree fit file
- `bayescycle plot`: ArviZ NetCDF/DataTree fit file -> PNG/SVG plot

Both verbs inherit stdio from the underlying `bayesite-idata`/`bayesite-viz`
process, so an agent still sees the same path-only stdout contract for
`plot` (controlled by `-f/--format`).

## Boundary

```text
bayescycle run directory -> bayescycle idata -> fit.nc -> bayescycle plot -> image
```

`bayescycle plot` output is not canonical state and not a report database.
Images are visual evidence registered as artifacts.

## Invocation

```bash
bayescycle idata --help
bayescycle plot --help
```

There is no separate install step: `bayescycle idata`/`bayescycle plot` fetch
`bayesite-viz` through `uvx` against the pinned commit on first use. Only
`uv`/`uvx` need to be on `PATH`.

## Export a run directory

`bayescycle plot` auto-runs `idata` when `RUN_DIR/fit.nc` is missing, so an
explicit export step is only needed to control validation or output path:

```bash
bayescycle idata runs/fit-0001
bayescycle idata runs/fit-0001 -o runs/fit-0001/fit.nc --validate require
bayescycle idata runs/fit-0001 -o runs/fit-0001/fit.nc --validate require --engine /path/to/bayesite
```

Register `fit.nc` as a derived implementation artifact only if useful. It is
usually enough to register the visual outputs.

## Standard plots

Diagnostic plots:

```bash
bayescycle plot trace    runs/fit-0001 -o artifacts/fit-0001-trace.png
bayescycle plot rank     runs/fit-0001 -o artifacts/fit-0001-rank.png
bayescycle plot autocorr runs/fit-0001 -o artifacts/fit-0001-autocorr.png
bayescycle plot forest   runs/fit-0001 -o artifacts/fit-0001-forest.png
```

Energy diagnostics, when the fit contains `sample_stats.energy`:

```bash
bayescycle plot energies runs/fit-0001 -o artifacts/fit-0001-energies.png
```

Posterior and pair plots:

```bash
bayescycle plot posterior runs/fit-0001 --kind hist -o artifacts/fit-0001-posterior.png
bayescycle plot pair      runs/fit-0001 --var mu --var tau -o artifacts/fit-0001-pair.png
```

Posterior predictive checks:

```bash
bayescycle plot ppc runs/fit-0001 --kind dist      -o artifacts/fit-0001-ppc-dist.png
bayescycle plot ppc runs/fit-0001 --kind interval  -o artifacts/fit-0001-ppc-interval.png
bayescycle plot ppc runs/fit-0001 --kind pit       -o artifacts/fit-0001-ppc-pit.png
bayescycle plot ppc runs/fit-0001 --kind rootogram -o artifacts/fit-0001-ppc-rootogram.png
bayescycle plot ppc runs/fit-0001 --kind tstat     -o artifacts/fit-0001-ppc-tstat.png
```

`rootogram` is for discrete outcomes. Choose PPC kinds based on the approved
estimator plan and model criticism questions.

## Shared options

Useful options from the `bayescycle plot` contract:

```text
-o FILE                       output path
--svg                         emit SVG instead of PNG
-f path|markdown|html|json|alt bayesite-viz stdout announce format
--var NAME                    repeatable variable selection
--coords key=val              repeatable coordinate selection
-b matplotlib|bokeh|plotly    backend; matplotlib is the v1 default/tested backend
--fit FIT_PATH                fit .nc path override (default: RUN_DIR/fit.nc)
--no-auto-idata                require an existing fit.nc instead of auto-exporting
```

## Workflow use

At phase gates, register `bayescycle plot` outputs as visual artifacts:

- `fit_diagnostic_visual_report`: trace/rank/autocorr/forest/energies
- `posterior_estimand_visual_report`: posterior plot for approved estimand
- `posterior_predictive_visual_report`: PPC plots
- `sensitivity_visual_report`: comparison plots across variants

A phase report should include the exact commands used and human interpretation
of what the plots show. The agent may summarize, but the human approves the
visual interpretation.

## Public contracts used

- `bayescycle idata <run-dir>` for run-directory export
- `bayescycle plot <verb> <run-dir>` for plots
- stdout path-only output contract for agent capture

## Forbidden assumptions

- Do not make `bayescycle plot` discover run directories other than the one
  named on the command line.
- Do not parse Bayesite NDJSON inside plotting commands; use the `idata`
  exporter boundary.
- Do not treat visual artifacts as approval; they are evidence for human review.
- If this adapter conflicts with `bayesite-viz` invariants, the `bayesite-viz`
  repository wins and the agent must stop and ask. Invoking `bayesite-idata`/
  `bayesite-viz` directly (bypassing `bayescycle idata`/`bayescycle plot`) is
  an escape hatch for debugging the exporter/plotter themselves, not the
  default workflow path.

## Failure handling

If `bayescycle idata` cannot export the run, do not skip visualization silently.
Record a diagnostics or visualization artifact with the failure and ask whether
to:

1. fix the run/export problem,
2. use an alternate plotting script as a temporary adapter, or
3. waive the visual gate for this phase.
