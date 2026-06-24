# Adapter: bayesite-viz

`bayesite-viz` is the visualization boundary for bayescycle studies.

Repository: <https://github.com/StefanSko/bayesite-viz>

It provides two agent-operable CLIs:

- `bayesite-idata`: Bayesite run directory -> ArviZ NetCDF/DataTree fit file
- `bayesite-viz`: ArviZ NetCDF/DataTree fit file -> PNG/SVG plot

The plot command prints only the absolute output path to stdout by default, so
agents can capture and register visual artifacts deterministically.

## Boundary

```text
bayescycle/Bayesite run directory -> bayesite-idata -> fit.nc -> bayesite-viz -> image
```

`bayesite-viz` is not canonical state and not a report database. Images are
visual evidence registered as artifacts.

## Install / invocation

Preferred installed commands:

```bash
bayesite-idata --help
bayesite-viz --help
```

Ad hoc invocation:

```bash
uvx bayesite-viz --help
```

## Export a run directory

Before plotting, export the run directory to ArviZ NetCDF:

```bash
bayesite-idata runs/fit-0001 -o runs/fit-0001/fit.nc
```

Validation options when available:

```bash
bayesite-idata runs/fit-0001 -o runs/fit-0001/fit.nc --validate require
bayesite-idata runs/fit-0001 -o runs/fit-0001/fit.nc --validate require --bayesite /path/to/bayesite
```

Register `fit.nc` as a derived implementation artifact only if useful. It is
usually enough to register the visual outputs.

## Standard plots

Diagnostic plots:

```bash
bayesite-viz trace    runs/fit-0001/fit.nc -o artifacts/fit-0001-trace.png
bayesite-viz rank     runs/fit-0001/fit.nc -o artifacts/fit-0001-rank.png
bayesite-viz autocorr runs/fit-0001/fit.nc -o artifacts/fit-0001-autocorr.png
bayesite-viz forest   runs/fit-0001/fit.nc -o artifacts/fit-0001-forest.png
```

Energy diagnostics, when the fit contains `sample_stats.energy`:

```bash
bayesite-viz energies runs/fit-0001/fit.nc -o artifacts/fit-0001-energies.png
```

Posterior and pair plots:

```bash
bayesite-viz posterior runs/fit-0001/fit.nc --kind hist -o artifacts/fit-0001-posterior.png
bayesite-viz pair      runs/fit-0001/fit.nc --var mu --var tau -o artifacts/fit-0001-pair.png
```

Posterior predictive checks:

```bash
bayesite-viz ppc runs/fit-0001/fit.nc --kind dist      -o artifacts/fit-0001-ppc-dist.png
bayesite-viz ppc runs/fit-0001/fit.nc --kind interval  -o artifacts/fit-0001-ppc-interval.png
bayesite-viz ppc runs/fit-0001/fit.nc --kind pit       -o artifacts/fit-0001-ppc-pit.png
bayesite-viz ppc runs/fit-0001/fit.nc --kind rootogram -o artifacts/fit-0001-ppc-rootogram.png
bayesite-viz ppc runs/fit-0001/fit.nc --kind tstat     -o artifacts/fit-0001-ppc-tstat.png
```

`rootogram` is for discrete outcomes. Choose PPC kinds based on the approved
estimator plan and model criticism questions.

## Shared options

Useful options from the bayesite-viz contract:

```text
-o FILE                       output path
--svg                         emit SVG instead of PNG
-f path|markdown|html|json|alt stdout format
--alt TEXT                    alt text for non-path formats
--var NAME                    repeatable variable selection
--like / --regex              variable selection mode
--coords key=val              repeatable coordinate selection
-b matplotlib|bokeh|plotly    backend; matplotlib is the v1 default/tested backend
```

## Workflow use

At phase gates, register bayesite-viz outputs as visual artifacts:

- `fit_diagnostic_visual_report`: trace/rank/autocorr/forest/energies
- `posterior_estimand_visual_report`: posterior plot for approved estimand
- `posterior_predictive_visual_report`: PPC plots
- `sensitivity_visual_report`: comparison plots across variants

A phase report should include the exact commands used and human interpretation
of what the plots show. The agent may summarize, but the human approves the
visual interpretation.

## Failure handling

If `bayesite-idata` cannot export the run, do not skip visualization silently.
Record a diagnostics or visualization artifact with the failure and ask whether
to:

1. fix the run/export problem,
2. use an alternate plotting script as a temporary adapter, or
3. waive the visual gate for this phase.
