"""uvx-mediated invocation of the bayesite-viz CLIs (bayesite-idata, bayesite-viz).

`bayesite-viz` and `bayesite-idata` ship as installable PyPI packages, so
bayescycle reaches them the same way an agent would from the command line:
`uvx --from <source> <entry-point> ...`. This module owns the pins
(`BAYESITE_VIZ_SOURCE`, `BAYESITE_IDATA_SOURCE`, `BAYESITE_VIZ_EXCLUDE_NEWER`)
and builds the argv for both entry points as pure, testable functions before
any subprocess runs. It also builds the argv for a "warmup" invocation of
each entry point (`--help`, which is enough for `uvx` to resolve and cache
the environment) so offline users can pre-materialize both uvx environments
and fail early instead of mid-workflow.

Two pins move together on every lockstep release, both stamped only by
scripts/bump_version.py: the version pin (duplicated here as
`BAYESITE_VIZ_SOURCE` and `BAYESITE_IDATA_SOURCE`, one per PyPI
distribution) and the timestamp pin (`BAYESITE_VIZ_EXCLUDE_NEWER`).
`FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION` is not one of the moving pins -- it is
a fixed far-future constant explained where it is defined below.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from bayescycle._errors import WorkflowError
from bayescycle._integrations.external_command import ExternalCommand, run_external_command

# Consumer pin: exact bayesite-viz PyPI version. Bump deliberately via
# scripts/bump_version.py alongside a coordinated compatibility review --
# this is the single place the pin lives (see module docstring: "two pins").
BAYESITE_VIZ_SOURCE = "bayesite-viz==0.7.0"

# Consumer pin: exact bayesite-idata PyPI version, moved in lockstep with
# BAYESITE_VIZ_SOURCE by scripts/bump_version.py.
BAYESITE_IDATA_SOURCE = "bayesite-idata==0.7.0"

# Determinism pin: the transitive-dependency half of the two source pins
# above. `uvx` resolves bayesite-viz/bayesite-idata's own dependencies
# (arviz, matplotlib, netcdf4, xarray, ...) as `>=`-range requirements at
# invocation time, so without a resolution cutoff an upstream release can
# change what gets installed -- and can break `bayescycle idata`/`plot` --
# with no change on our side. `--exclude-newer` freezes resolution to
# package versions published on or before this timestamp, making the
# plot/idata environments a pure function of the pins together. Stamped by
# scripts/bump_version.py to the moment of the version bump.
BAYESITE_VIZ_EXCLUDE_NEWER = "2026-07-20T17:55:54Z"

# The first-party packages above are exact version pins (`==X.Y.Z`), so a
# resolver can only ever pick that one version of them -- the global
# `--exclude-newer` cutoff cannot cause version drift for them the way it
# can for `>=`-range transitive dependencies. But `--exclude-newer` is
# evaluated *before* pins narrow the candidate set, so without this
# exemption the just-published first-party wheel (dated after
# BAYESITE_VIZ_EXCLUDE_NEWER was stamped) would itself be invisible to the
# resolver. `--exclude-newer-package <name>=<this value>` overrides the
# cutoff for that one package only, so the first-party wheel resolves while
# every transitive dependency still resolves frozen at the bump-time
# cutoff. A fixed far-future constant, never rewritten by
# scripts/bump_version.py.
FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION = "2100-01-01T00:00:00Z"


def _base_argv(source: str, package_name: str) -> list[str]:
    """Build the shared uvx flags/`--from` prefix for one bayesite-viz entry point."""
    return [
        "uvx",
        "--quiet",
        "--exclude-newer",
        BAYESITE_VIZ_EXCLUDE_NEWER,
        "--exclude-newer-package",
        f"{package_name}={FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION}",
        "--from",
        source,
    ]


# The nine bayesite-viz plot verbs, in the order the bayesite-viz CLI group
# registers them.
VIZ_VERBS: tuple[str, ...] = (
    "trace",
    "rank",
    "forest",
    "energies",
    "pair",
    "posterior",
    "autocorr",
    "ess-rhat",
    "ppc",
)

_KIND_VERBS = frozenset({"posterior", "ppc"})


@dataclass(frozen=True)
class IdataOptions:
    """Options forwarded to `bayesite-idata`."""

    run_dir: Path
    output: Path
    validate: str | None = None
    bayesite: str | None = None


@dataclass(frozen=True)
class PlotOptions:
    """Options forwarded to one `bayesite-viz` verb."""

    verb: str
    fit_path: Path
    output: Path | None = None
    kind: str | None = None
    fmt: str | None = None
    variables: tuple[str, ...] = ()
    coords: tuple[tuple[str, str], ...] = ()
    backend: str | None = None
    svg: bool = False


def default_fit_path(run_dir: Path) -> Path:
    """Return the conventional fit-file path for a run directory."""
    return run_dir / "fit.nc"


def idata_command(options: IdataOptions, *, source: str = BAYESITE_IDATA_SOURCE) -> ExternalCommand:
    """Build the argv for `bayesite-idata` without running anything."""
    argv: list[str] = _base_argv(source, "bayesite-idata")
    argv.extend(
        (
            "bayesite-idata",
            str(options.run_dir),
            "-o",
            str(options.output),
        )
    )
    if options.validate is not None:
        argv.extend(("--validate", options.validate))
    if options.bayesite is not None:
        argv.extend(("--bayesite", options.bayesite))
    return ExternalCommand(argv=tuple(argv), output_paths=(options.output,))


def plot_command(options: PlotOptions, *, source: str = BAYESITE_VIZ_SOURCE) -> ExternalCommand:
    """Build the argv for one `bayesite-viz` verb without running anything."""
    if options.verb not in VIZ_VERBS:
        known = ", ".join(VIZ_VERBS)
        raise WorkflowError(
            f"unknown bayesite-viz verb: {options.verb!r}; expected one of: {known}"
        )
    if options.kind is not None and options.verb not in _KIND_VERBS:
        raise WorkflowError(
            f"--kind is only supported for the posterior and ppc verbs, not {options.verb!r}"
        )
    argv: list[str] = _base_argv(source, "bayesite-viz")
    argv.extend(("bayesite-viz", options.verb, str(options.fit_path)))
    if options.output is not None:
        argv.extend(("-o", str(options.output)))
    if options.kind is not None:
        argv.extend(("--kind", options.kind))
    if options.fmt is not None:
        argv.extend(("-f", options.fmt))
    for variable in options.variables:
        argv.extend(("--var", variable))
    for key, value in options.coords:
        argv.extend(("--coords", f"{key}={value}"))
    if options.backend is not None:
        argv.extend(("-b", options.backend))
    if options.svg:
        argv.append("--svg")
    output_paths = (options.output,) if options.output is not None else ()
    return ExternalCommand(argv=tuple(argv), output_paths=output_paths)


def idata_warmup_command(*, source: str = BAYESITE_IDATA_SOURCE) -> ExternalCommand:
    """Build the argv that pre-materializes the `bayesite-idata` uvx environment.

    `--help` runs no real work but is enough for `uvx` to resolve and cache
    the environment against the same `--from` source and `--exclude-newer`
    cutoff `idata_command` would use.
    """
    argv = _base_argv(source, "bayesite-idata")
    argv.extend(("bayesite-idata", "--help"))
    return ExternalCommand(argv=tuple(argv))


def plot_warmup_command(*, source: str = BAYESITE_VIZ_SOURCE) -> ExternalCommand:
    """Build the argv that pre-materializes the `bayesite-viz` uvx environment.

    `--help` runs no real work but is enough for `uvx` to resolve and cache
    the environment against the same `--from` source and `--exclude-newer`
    cutoff `plot_command` would use.
    """
    argv = _base_argv(source, "bayesite-viz")
    argv.extend(("bayesite-viz", "--help"))
    return ExternalCommand(argv=tuple(argv))


def warmup_commands(
    *,
    idata_source: str = BAYESITE_IDATA_SOURCE,
    viz_source: str = BAYESITE_VIZ_SOURCE,
) -> tuple[ExternalCommand, ExternalCommand]:
    """Return both entry points' warmup commands, idata first then viz."""
    return (
        idata_warmup_command(source=idata_source),
        plot_warmup_command(source=viz_source),
    )


def run_idata(options: IdataOptions, *, source: str = BAYESITE_IDATA_SOURCE) -> int:
    """Run `bayesite-idata` under uvx, inheriting stdio."""
    _require_uvx()
    return run_external_command(idata_command(options, source=source))


def run_plot(options: PlotOptions, *, source: str = BAYESITE_VIZ_SOURCE) -> int:
    """Run one `bayesite-viz` verb under uvx, inheriting stdio."""
    _require_uvx()
    return run_external_command(plot_command(options, source=source))


def run_idata_warmup(*, source: str = BAYESITE_IDATA_SOURCE) -> int:
    """Pre-materialize the `bayesite-idata` uvx environment, inheriting stdio."""
    _require_uvx()
    return run_external_command(idata_warmup_command(source=source))


def run_plot_warmup(*, source: str = BAYESITE_VIZ_SOURCE) -> int:
    """Pre-materialize the `bayesite-viz` uvx environment, inheriting stdio."""
    _require_uvx()
    return run_external_command(plot_warmup_command(source=source))


def run_warmup(
    *,
    idata_source: str = BAYESITE_IDATA_SOURCE,
    viz_source: str = BAYESITE_VIZ_SOURCE,
) -> int:
    """Pre-materialize both bayesite-viz uvx environments, inheriting stdio.

    Runs the idata warmup first and only proceeds to the viz warmup if it
    succeeds, so a broken/offline cache fails fast with one clear command's
    stdio rather than interleaving both.
    """
    _require_uvx()
    idata_code = run_external_command(idata_warmup_command(source=idata_source))
    if idata_code != 0:
        return idata_code
    return run_external_command(plot_warmup_command(source=viz_source))


def _require_uvx() -> None:
    if shutil.which("uvx") is None:
        raise WorkflowError(
            "uvx was not found on PATH; bayescycle idata/plot run bayesite-viz through uvx.\n\n"
            "Install uv (which provides uvx): "
            "https://docs.astral.sh/uv/getting-started/installation/"
        )
