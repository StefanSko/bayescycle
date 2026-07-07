"""CLI: one Click command per verb, shared output contract."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import click

from bayesite_viz import plots
from bayesite_viz.io import load_fit, require_groups, require_var
from bayesite_viz.output import (
    OUTPUT_FORMATS,
    OutputSpec,
    announce,
    auto_alt,
    output_format,
    resolve_output_path,
)


def _coords(
    ctx: click.Context, param: click.Parameter, value: tuple[str, ...]
) -> plots.CoordSelection | None:
    if not value:
        return None
    out: plots.CoordSelection = {}
    for item in value:
        if "=" not in item:
            msg = f"--coords expects key=value, got {item!r}"
            raise click.BadParameter(msg, ctx=ctx, param=param)
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _shared_options[FuncT: Callable[..., object]](f: FuncT) -> FuncT:
    # Click's option() decorators take and return arbitrary callables.
    f = click.option(
        "-o", "--output", "output", default=None, help="Output path; overwrites if it exists."
    )(f)
    f = click.option(
        "-f",
        "--output-format",
        "fmt",
        type=click.Choice(OUTPUT_FORMATS),
        default="path",
        help="What to print to stdout.",
    )(f)
    f = click.option("--alt", "alt", default=None, help="Alt text (auto-generated when omitted).")(
        f
    )
    f = click.option("--var", "var", multiple=True, help="Variable name; repeatable.")(f)
    f = click.option(
        "--like",
        "filter_vars",
        flag_value="like",
        default=None,
        help="Treat --var as a substring pattern (like).",
    )(f)
    f = click.option(
        "--regex",
        "filter_vars",
        flag_value="regex",
        default=None,
        help="Treat --var as a regex pattern.",
    )(f)
    f = click.option(
        "--coords", "coords", multiple=True, callback=_coords, help="coords key=val; repeatable."
    )(f)
    f = click.option(
        "-b",
        "--backend",
        "backend",
        type=click.Choice(["matplotlib", "bokeh", "plotly"]),
        default=plots.DEFAULT_BACKEND,
        show_default=True,
    )(f)
    f = click.option("--svg", "svg", is_flag=True, default=False, help="Emit SVG instead of PNG.")(
        f
    )
    return f


def _run(
    verb: str,
    fit: Path,
    plot_fn: plots.PlotFn,
    *,
    output: str | None,
    fmt: str,
    alt: str | None,
    var: tuple[str, ...],
    filter_vars: str | None,
    coords: plots.CoordSelection | None,
    backend: str,
    svg: bool,
    style_kind: str | None = None,
    required_groups: tuple[str, ...] = (),
    required_vars: tuple[tuple[str, str], ...] = (),
    kind_for_alt: str | None = None,
) -> None:
    dt = load_fit(fit)
    if required_groups:
        require_groups(dt, *required_groups)
    for group, vname in required_vars:
        require_var(dt, group, vname)
    args = plots.PlotArgs(
        var_names=var or None,
        filter_vars=plots.filter_mode(filter_vars),
        coords=coords,
        backend=backend,
        style_kind=plots.dist_kind(style_kind),
    )
    try:
        pc = plot_fn(dt, args)
    except (ValueError, KeyError) as e:
        sys.stderr.write(f"{e}\n")
        raise SystemExit(4) from e
    except (ImportError, ModuleNotFoundError) as e:
        sys.stderr.write(
            f"backend {backend!r} is not available: {e}. "
            f"Install the backend's dependencies (e.g. `uv pip install bokeh`) "
            f"or use --backend matplotlib.\n"
        )
        raise SystemExit(6) from e
    ext = "svg" if svg else "png"
    if output is not None:
        given_suffix = Path(output).suffix.lower().lstrip(".")
        if given_suffix != ext:
            sys.stderr.write(
                f"output path {output!r} has suffix .{given_suffix or '(none)'} but "
                f"{'--svg is set' if svg else 'PNG is the default format'}; "
                f"use .{ext} or {'drop --svg' if svg else 'pass --svg'}.\n"
            )
            raise SystemExit(2)
    path = resolve_output_path(verb, output, ext)
    try:
        pc.savefig(path)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError, RuntimeError) as e:
        sys.stderr.write(
            f"backend {backend!r} cannot save {ext}: {e}. "
            f"Install the backend's export dependencies or use --backend matplotlib.\n"
        )
        raise SystemExit(6) from e
    # Release the matplotlib figure so repeated calls don't leak/accumulate.
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close("all")
    resolved_alt = alt if alt is not None else auto_alt(verb, fit, var or None, kind_for_alt)
    spec = OutputSpec(path=path, fmt=output_format(fmt), alt=resolved_alt)
    announce(spec)


def _fit_arg[FuncT: Callable[..., object]]() -> Callable[[FuncT], FuncT]:
    return click.argument("fit", type=click.Path(path_type=Path))


# --- 1:1 verbs --------------------------------------------------------------


@click.group()
def cli() -> None:
    """Render ArviZ Bayesian plots from an InferenceData fit."""


def _make_1to1(spec: plots.VerbSpec) -> None:
    @cli.command(name=spec.name)
    @_fit_arg()
    @_shared_options
    def _cmd(
        fit: Path,
        output: str | None,
        fmt: str,
        alt: str | None,
        var: tuple[str, ...],
        filter_vars: str | None,
        coords: plots.CoordSelection | None,
        backend: str,
        svg: bool,
    ) -> None:
        _run(
            spec.name,
            fit,
            spec.plot_fn,
            output=output,
            fmt=fmt,
            alt=alt,
            var=var,
            filter_vars=filter_vars,
            coords=coords,
            backend=backend,
            svg=svg,
            required_groups=spec.required_groups,
            required_vars=spec.required_vars,
        )


for _spec in plots.VERBS_1TO1:
    _make_1to1(_spec)


# --- ess-rhat (McElreath first-look diagnostic panel) ----------------------


@cli.command(name="ess-rhat")
@_fit_arg()
@_shared_options
def _ess_rhat(
    fit: Path,
    output: str | None,
    fmt: str,
    alt: str | None,
    var: tuple[str, ...],
    filter_vars: str | None,
    coords: plots.CoordSelection | None,
    backend: str,
    svg: bool,
) -> None:
    _run(
        "ess-rhat",
        fit,
        plots.ess_rhat,
        output=output,
        fmt=fmt,
        alt=alt,
        var=var,
        filter_vars=filter_vars,
        coords=coords,
        backend=backend,
        svg=svg,
        required_groups=plots.ESS_RHAT_VERB.required_groups,
    )


# --- posterior (plot_dist with a style --kind) ----------------------------


@cli.command(name="posterior")
@_fit_arg()
@_shared_options
@click.option(
    "--kind",
    "kind",
    type=click.Choice(list(plots.DIST_KINDS)),
    default=None,
    help="Posterior marginal style: auto | kde | hist | dot | ecdf.",
)
def _posterior(
    fit: Path,
    output: str | None,
    fmt: str,
    alt: str | None,
    var: tuple[str, ...],
    filter_vars: str | None,
    coords: plots.CoordSelection | None,
    backend: str,
    svg: bool,
    kind: str | None,
) -> None:
    _run(
        "posterior",
        fit,
        plots.posterior,
        output=output,
        fmt=fmt,
        alt=alt,
        var=var,
        filter_vars=filter_vars,
        coords=coords,
        backend=backend,
        svg=svg,
        style_kind=kind,
        required_groups=plots.POSTERIOR_VERB.required_groups,
    )


# --- ppc (family dispatch) --------------------------------------------------


@cli.command(name="ppc")
@_fit_arg()
@_shared_options
@click.option(
    "--kind",
    "kind",
    type=click.Choice(list(plots.PPC_KINDS)),
    default=plots.DEFAULT_PPC_KIND,
    show_default=True,
    help="PPC plot family; selects the arviz_plots function.",
)
def _ppc(
    fit: Path,
    output: str | None,
    fmt: str,
    alt: str | None,
    var: tuple[str, ...],
    filter_vars: str | None,
    coords: plots.CoordSelection | None,
    backend: str,
    svg: bool,
    kind: str,
) -> None:
    _run(
        "ppc",
        fit,
        lambda dt, args: plots.ppc(dt, args, kind=plots.ppc_kind(kind)),
        output=output,
        fmt=fmt,
        alt=alt,
        var=var,
        filter_vars=filter_vars,
        coords=coords,
        backend=backend,
        svg=svg,
        required_groups=plots.PPC_VERB.required_groups,
        kind_for_alt=kind,
    )
