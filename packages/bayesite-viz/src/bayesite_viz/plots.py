"""Per-verb arviz_plots wrappers.

Each 1:1 verb is one thin function that calls exactly one ``arviz_plots``
function. ``ppc`` is the sole family-dispatching verb: ``kind`` selects the
function. ``ess-rhat`` is the diagnostic-composition exception, delegating ESS
and Rhat computation to ``arviz-stats`` and arranging the result with
``arviz_plots`` primitives. Validated against arviz-plots 1.2.0 / arviz-base
1.2.0 / arviz-stats 1.2.0.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import arviz_plots as azp
import arviz_stats as azs
import xarray as xr
from arviz_plots import PlotCollection
from arviz_plots.visuals import (
    hline,
    labelled_title,
    labelled_x,
    labelled_y,
    scatter_xy,
    set_ylim,
    set_yscale,
    vline,
)

type FilterMode = Literal["like", "regex"]
type DistKind = Literal["auto", "kde", "hist", "dot", "ecdf"]
type PpcKind = Literal["dist", "interval", "pit", "rootogram", "tstat"]
type CoordSelection = dict[str, str]
type PlotCallable = Callable[..., PlotCollection]


class ArvizPlotKwargs(TypedDict, total=False):
    """Keyword subset bayesite-viz forwards into arviz_plots."""

    backend: str
    var_names: list[str]
    filter_vars: FilterMode
    coords: CoordSelection
    kind: DistKind


_FILTER_MODES_BY_VALUE: dict[str, FilterMode] = {"like": "like", "regex": "regex"}
_DIST_KINDS_BY_VALUE: dict[str, DistKind] = {
    "auto": "auto",
    "kde": "kde",
    "hist": "hist",
    "dot": "dot",
    "ecdf": "ecdf",
}
_PPC_KINDS_BY_VALUE: dict[str, PpcKind] = {
    "dist": "dist",
    "interval": "interval",
    "pit": "pit",
    "rootogram": "rootogram",
    "tstat": "tstat",
}

DEFAULT_BACKEND = "matplotlib"
_SAMPLE_DIMS: tuple[str, str] = ("chain", "draw")

# Verbs whose arviz_plots function reads posterior_predictive/observed_data.
_PPC_FAMILY = {
    "dist": azp.plot_ppc_dist,
    "interval": azp.plot_ppc_interval,
    "pit": azp.plot_ppc_pit,
    "rootogram": azp.plot_ppc_rootogram,
    "tstat": azp.plot_ppc_tstat,
}
PPC_KINDS: tuple[PpcKind, ...] = ("dist", "interval", "pit", "rootogram", "tstat")
DEFAULT_PPC_KIND: PpcKind = "dist"

# Verbs with a plot-style --kind (same function, different style).
DIST_KINDS: tuple[DistKind, ...] = ("auto", "kde", "hist", "dot", "ecdf")


def filter_mode(value: str | None) -> FilterMode | None:
    """Normalize a Click-validated variable filter mode."""
    if value is None:
        return None
    return _FILTER_MODES_BY_VALUE[value]


def dist_kind(value: str | None) -> DistKind | None:
    """Normalize a Click-validated posterior plot style kind."""
    if value is None:
        return None
    return _DIST_KINDS_BY_VALUE[value]


def ppc_kind(value: str) -> PpcKind:
    """Normalize a Click-validated PPC function-dispatch kind."""
    return _PPC_KINDS_BY_VALUE[value]


@dataclass(frozen=True)
class PlotArgs:
    """Shared knobs an agent would not know to pass to arviz_plots."""

    var_names: tuple[str, ...] | None = None
    filter_vars: FilterMode | None = None
    coords: CoordSelection | None = None
    backend: str = DEFAULT_BACKEND
    style_kind: DistKind | None = None


type PlotFn = Callable[[xr.DataTree, PlotArgs], PlotCollection]
type RequiredVar = tuple[str, str]


@dataclass(frozen=True)
class VerbSpec:
    """One CLI verb's plot function and fit requirements."""

    name: str
    plot_fn: PlotFn
    required_groups: tuple[str, ...] = ()
    required_vars: tuple[RequiredVar, ...] = ()


def _kw(args: PlotArgs) -> ArvizPlotKwargs:
    kw: ArvizPlotKwargs = {"backend": args.backend}
    if args.var_names:
        kw["var_names"] = list(args.var_names)
    if args.filter_vars:
        kw["filter_vars"] = args.filter_vars
    if args.coords:
        kw["coords"] = args.coords
    if args.style_kind:
        kw["kind"] = args.style_kind
    return kw


def _call_plot(fn: PlotCallable, dt: xr.DataTree, kwargs: ArvizPlotKwargs) -> PlotCollection:
    """Call one arviz_plots function through the typed forwarded-keyword seam."""
    return fn(dt, **kwargs)


# --- 1:1 verbs --------------------------------------------------------------


def trace(dt: xr.DataTree, args: PlotArgs) -> PlotCollection:
    """plot_trace."""
    return _call_plot(azp.plot_trace, dt, _kw(args))


def rank(dt: xr.DataTree, args: PlotArgs) -> PlotCollection:
    """plot_rank (uniform reference interval built-in)."""
    return _call_plot(azp.plot_rank, dt, _kw(args))


def forest(dt: xr.DataTree, args: PlotArgs) -> PlotCollection:
    """plot_forest (no kind; ridge is a separate function)."""
    return _call_plot(azp.plot_forest, dt, _kw(args))


def energies(dt: xr.DataTree, args: PlotArgs) -> PlotCollection:
    """plot_energy (NUTS energy / BFMI). Reads sample_stats.

    plot_energy does not accept var_names/filter_vars/coords as selectors, so
    those shared flags are deliberately not forwarded.
    """
    kw: ArvizPlotKwargs = {"backend": args.backend}
    if args.style_kind:
        kw["kind"] = args.style_kind
    return _call_plot(azp.plot_energy, dt, kw)


def pair(dt: xr.DataTree, args: PlotArgs) -> PlotCollection:
    """plot_pair. Requires var_names: many-parameter models exceed the subplot cap."""
    if not args.var_names:
        msg = "pair requires --var; plotting all variables can exceed the subplot cap"
        raise ValueError(msg)
    return _call_plot(azp.plot_pair, dt, _kw(args))


def posterior(dt: xr.DataTree, args: PlotArgs) -> PlotCollection:
    """plot_dist (posterior marginal). --kind is a style flag, not function dispatch."""
    return _call_plot(azp.plot_dist, dt, _kw(args))


def autocorr(dt: xr.DataTree, args: PlotArgs) -> PlotCollection:
    """plot_autocorr."""
    return _call_plot(azp.plot_autocorr, dt, _kw(args))


def ess_rhat(dt: xr.DataTree, args: PlotArgs) -> PlotCollection:
    """McElreath-style first-look scatter: bulk ESS vs rank-normalized Rhat.

    This is intentionally a small exception to the 1:1 arviz_plots wrapper rule:
    the diagnostics are computed by arviz-stats, then arranged with arviz_plots
    primitives so the CLI can reproduce rethinking::dashboard()'s first panel.
    """
    diagnostics = _ess_rhat_dataset(dt, args)
    values = diagnostics["diagnostics"]
    sample_count = _posterior_sample_count(dt)

    pc = PlotCollection.wrap(diagnostics, cols=[], backend=args.backend)
    pc.map(scatter_xy, "points", data=values)
    pc.map(
        vline, "ess_warning", data=xr.DataArray(0.1 * sample_count, name="ess_warning"), color="red"
    )
    pc.map(vline, "raw_samples", data=xr.DataArray(sample_count, name="raw_samples"), color="gray")
    pc.map(hline, "rhat_one", data=xr.DataArray(1.0, name="rhat_one"), linestyle="dashed")
    pc.map(set_yscale, "yscale", data=values, scale="log")
    pc.map(set_ylim, "ylim", data=values, limits=_rhat_ylim(values))
    pc.map(labelled_x, "xlabel", data=values, text="bulk effective sample size")
    pc.map(labelled_y, "ylabel", data=values, text="Rhat")
    pc.map(labelled_title, "title", data=values, text="ESS bulk vs Rhat")
    return pc


def _ess_rhat_dataset(dt: xr.DataTree, args: PlotArgs) -> xr.Dataset:
    var_names = list(args.var_names) if args.var_names else None
    ess_dt = cast(
        xr.DataTree,
        azs.ess(
            dt,
            group="posterior",
            var_names=var_names,
            filter_vars=args.filter_vars,
            coords=args.coords,
            sample_dims=_SAMPLE_DIMS,
            method="bulk",
        ),
    )
    rhat_dt = cast(
        xr.DataTree,
        azs.rhat(
            dt,
            group="posterior",
            var_names=var_names,
            filter_vars=args.filter_vars,
            coords=args.coords,
            sample_dims=_SAMPLE_DIMS,
            method="rank",
        ),
    )
    ess_values = _flatten_diagnostic(ess_dt["/posterior"].ds, "ESS")
    rhat_values = _flatten_diagnostic(rhat_dt["/posterior"].ds, "Rhat")
    ess_aligned, rhat_aligned = xr.align(ess_values, rhat_values, join="inner")
    if ess_aligned.sizes.get("label", 0) == 0:
        msg = "no posterior variables selected for ess-rhat"
        raise ValueError(msg)
    values = xr.concat(
        (ess_aligned, rhat_aligned), dim=xr.IndexVariable("plot_axis", ["x", "y"])
    ).transpose("label", "plot_axis")
    values.name = "diagnostics"
    return xr.Dataset({"diagnostics": values})


def _flatten_diagnostic(ds: xr.Dataset, diagnostic_name: str) -> xr.DataArray:
    if not ds.data_vars:
        msg = f"no posterior variables available for {diagnostic_name}"
        raise ValueError(msg)
    values = ds.to_stacked_array("label", sample_dims=[]).astype(float)
    if values.sizes.get("label", 0) == 0:
        msg = f"no posterior variables available for {diagnostic_name}"
        raise ValueError(msg)
    return values


def _posterior_sample_count(dt: xr.DataTree) -> float:
    posterior = dt["/posterior"].ds
    missing = tuple(dim for dim in _SAMPLE_DIMS if dim not in posterior.sizes)
    if missing:
        msg = f"ess-rhat requires posterior sample dimension(s): {', '.join(missing)}"
        raise ValueError(msg)
    return float(posterior.sizes["chain"] * posterior.sizes["draw"])


def _rhat_ylim(values: xr.DataArray) -> tuple[float, float]:
    rhat_values = values.sel(plot_axis="y").values.reshape(-1)
    finite = [float(value) for value in rhat_values if math.isfinite(float(value))]
    if not finite:
        return (1.0, 1.1)
    positive = [value for value in finite if value > 0]
    if not positive:
        return (1.0, 1.1)
    lower = min(1.0, min(positive))
    upper = max(1.1, max(positive))
    if lower == upper:
        upper = lower * 1.1
    return (lower, upper)


# --- family-dispatching verb ------------------------------------------------


def ppc(dt: xr.DataTree, args: PlotArgs, kind: PpcKind = DEFAULT_PPC_KIND) -> PlotCollection:
    """Dispatch across the plot_ppc_* family by kind.

    kind is function dispatch, not plot style: dist | interval | pit |
    rootogram | tstat. rootogram is discrete-only.
    """
    if kind not in _PPC_FAMILY:
        msg = f"unknown ppc kind: {kind}; choose from {', '.join(PPC_KINDS)}"
        raise ValueError(msg)
    if kind != "tstat":
        _reject_scalar_observed_ppc(dt, args)
    fn = _PPC_FAMILY[kind]
    return _call_plot(fn, dt, _kw(args))


def _reject_scalar_observed_ppc(dt: xr.DataTree, args: PlotArgs) -> None:
    observed = dt["/observed_data"].ds
    scalar_names = tuple(
        name
        for name in _selected_ppc_variables(dt, args)
        if _observed_after_coords(observed, name, args.coords).ndim == 0
    )
    if not scalar_names:
        return
    if len(scalar_names) == 1:
        msg = (
            f"ppc cannot render scalar observed variable {scalar_names[0]!r} with the current "
            "ArviZ backend; use a vector observed variable or a non-PPC plot until scalar "
            "PPC support is added"
        )
        raise ValueError(msg)
    names = ", ".join(repr(name) for name in scalar_names)
    msg = (
        f"ppc cannot render scalar observed variables {names} with the current ArviZ backend; "
        "use vector observed variables or a non-PPC plot until scalar PPC support is added"
    )
    raise ValueError(msg)


def _observed_after_coords(
    observed: xr.Dataset, name: str, coords: CoordSelection | None
) -> xr.DataArray:
    arr = observed[name]
    if not coords:
        return arr
    selection = {dim: value for dim, value in coords.items() if dim in arr.dims}
    if not selection:
        return arr
    try:
        return arr.sel(selection)
    except (KeyError, TypeError, ValueError):
        return arr


def _selected_ppc_variables(dt: xr.DataTree, args: PlotArgs) -> tuple[str, ...]:
    observed_names = {str(name) for name in dt["/observed_data"].ds.data_vars}
    predictive_names = {str(name) for name in dt["/posterior_predictive"].ds.data_vars}
    candidates = tuple(sorted(observed_names & predictive_names))
    if not args.var_names:
        return candidates
    include, exclude = _split_var_patterns(args.var_names, candidates)
    if exclude:
        selected = set(candidates)
    elif include:
        selected = set(_matching_names(candidates, include, args.filter_vars))
    else:
        selected = set(candidates)
    selected.difference_update(_matching_names(candidates, exclude, args.filter_vars))
    return tuple(name for name in candidates if name in selected)


def _split_var_patterns(
    var_names: tuple[str, ...], candidates: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    include: list[str] = []
    exclude: list[str] = []
    candidate_names = set(candidates)
    for name in var_names:
        if name.startswith("~") and name not in candidate_names:
            exclude.append(name[1:])
        else:
            include.append(name)
    return tuple(include), tuple(exclude)


def _matching_names(
    candidates: tuple[str, ...], patterns: tuple[str, ...], filter_vars: FilterMode | None
) -> tuple[str, ...]:
    if not patterns:
        return ()
    if filter_vars == "like":
        return tuple(name for name in candidates if any(pattern in name for pattern in patterns))
    if filter_vars == "regex":
        compiled = tuple(_compile_var_regex(pattern) for pattern in patterns)
        return tuple(
            name for name in candidates if any(pattern.search(name) for pattern in compiled)
        )
    exact = set(patterns)
    return tuple(name for name in candidates if name in exact)


def _compile_var_regex(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        msg = f"invalid --var regex {pattern!r}: {exc}"
        raise ValueError(msg) from exc


# Required groups are validated up front so missing fit contents produce clean
# repair messages (exit 3), not arviz internals tracebacks.
VERBS_1TO1: tuple[VerbSpec, ...] = (
    VerbSpec("trace", trace, required_groups=("posterior",)),
    VerbSpec("rank", rank, required_groups=("posterior",)),
    VerbSpec("forest", forest, required_groups=("posterior",)),
    VerbSpec(
        "energies",
        energies,
        required_groups=("sample_stats",),
        required_vars=(("sample_stats", "energy"),),
    ),
    VerbSpec("pair", pair, required_groups=("posterior",)),
    VerbSpec("autocorr", autocorr, required_groups=("posterior",)),
)
POSTERIOR_VERB = VerbSpec("posterior", posterior, required_groups=("posterior",))
ESS_RHAT_VERB = VerbSpec("ess-rhat", ess_rhat, required_groups=("posterior",))
PPC_VERB = VerbSpec("ppc", ppc, required_groups=("posterior_predictive", "observed_data"))
