"""Pure argv-construction tests for the bayesite-viz uvx runner.

`idata_command`/`plot_command` build argv without running anything, so these
tests exercise exact argv tuples and option forwarding without needing `uvx`
or network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bayescycle._errors import WorkflowError
from bayescycle.backends.bayesite_viz.uvx_runner import (
    BAYESITE_IDATA_SOURCE,
    BAYESITE_VIZ_EXCLUDE_NEWER,
    BAYESITE_VIZ_SOURCE,
    FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION,
    VIZ_VERBS,
    IdataOptions,
    PlotOptions,
    default_fit_path,
    idata_command,
    idata_warmup_command,
    plot_command,
    plot_warmup_command,
    warmup_commands,
)


def test_pinned_sources_are_exact_pypi_version_pins() -> None:
    """bayesite-viz and bayesite-idata are separate PyPI distributions now."""
    assert BAYESITE_VIZ_SOURCE.startswith("bayesite-viz==")
    assert BAYESITE_IDATA_SOURCE.startswith("bayesite-idata==")
    assert BAYESITE_VIZ_SOURCE != BAYESITE_IDATA_SOURCE


def test_default_fit_path_is_run_dir_slash_fit_nc() -> None:
    assert default_fit_path(Path("run")) == Path("run/fit.nc")
    assert default_fit_path(Path("/abs/run")) == Path("/abs/run/fit.nc")


def test_viz_verbs_match_the_bayesite_viz_contract() -> None:
    assert VIZ_VERBS == (
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


def test_idata_command_minimal() -> None:
    options = IdataOptions(run_dir=Path("run"), output=Path("run/fit.nc"))

    command = idata_command(options)

    assert command.argv == (
        "uvx",
        "--quiet",
        "--exclude-newer",
        BAYESITE_VIZ_EXCLUDE_NEWER,
        "--exclude-newer-package",
        f"bayesite-idata={FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION}",
        "--from",
        BAYESITE_IDATA_SOURCE,
        "bayesite-idata",
        "run",
        "-o",
        "run/fit.nc",
    )
    assert command.output_paths == (Path("run/fit.nc"),)


def test_idata_command_forwards_validate_and_bayesite() -> None:
    options = IdataOptions(
        run_dir=Path("run"),
        output=Path("run/fit.nc"),
        validate="require",
        bayesite="/opt/bayesite",
    )

    command = idata_command(options)

    assert command.argv == (
        "uvx",
        "--quiet",
        "--exclude-newer",
        BAYESITE_VIZ_EXCLUDE_NEWER,
        "--exclude-newer-package",
        f"bayesite-idata={FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION}",
        "--from",
        BAYESITE_IDATA_SOURCE,
        "bayesite-idata",
        "run",
        "-o",
        "run/fit.nc",
        "--validate",
        "require",
        "--bayesite",
        "/opt/bayesite",
    )


def test_idata_command_honors_source_override() -> None:
    options = IdataOptions(run_dir=Path("run"), output=Path("run/fit.nc"))

    command = idata_command(options, source="bayesite-idata==9.9.9")

    assert command.argv[7] == "bayesite-idata==9.9.9"


def test_plot_command_minimal() -> None:
    options = PlotOptions(verb="trace", fit_path=Path("run/fit.nc"))

    command = plot_command(options)

    assert command.argv == (
        "uvx",
        "--quiet",
        "--exclude-newer",
        BAYESITE_VIZ_EXCLUDE_NEWER,
        "--exclude-newer-package",
        f"bayesite-viz={FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION}",
        "--from",
        BAYESITE_VIZ_SOURCE,
        "bayesite-viz",
        "trace",
        "run/fit.nc",
    )
    assert command.output_paths == ()


def test_plot_command_forwards_all_set_options() -> None:
    options = PlotOptions(
        verb="posterior",
        fit_path=Path("run/fit.nc"),
        output=Path("viz/posterior.png"),
        kind="hist",
        fmt="json",
        variables=("mu", "tau"),
        coords=(("obs", "a"),),
        backend="bokeh",
        svg=True,
    )

    command = plot_command(options)

    assert command.argv == (
        "uvx",
        "--quiet",
        "--exclude-newer",
        BAYESITE_VIZ_EXCLUDE_NEWER,
        "--exclude-newer-package",
        f"bayesite-viz={FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION}",
        "--from",
        BAYESITE_VIZ_SOURCE,
        "bayesite-viz",
        "posterior",
        "run/fit.nc",
        "-o",
        "viz/posterior.png",
        "--kind",
        "hist",
        "-f",
        "json",
        "--var",
        "mu",
        "--var",
        "tau",
        "--coords",
        "obs=a",
        "-b",
        "bokeh",
        "--svg",
    )
    assert command.output_paths == (Path("viz/posterior.png"),)


def test_plot_command_honors_source_override() -> None:
    options = PlotOptions(verb="trace", fit_path=Path("run/fit.nc"))

    command = plot_command(options, source="bayesite-viz==9.9.9")

    assert command.argv[7] == "bayesite-viz==9.9.9"


def test_plot_command_rejects_unknown_verb() -> None:
    options = PlotOptions(verb="not-a-verb", fit_path=Path("run/fit.nc"))

    with pytest.raises(WorkflowError, match="unknown bayesite-viz verb"):
        plot_command(options)


def test_plot_command_rejects_kind_for_non_kind_verb() -> None:
    options = PlotOptions(verb="trace", fit_path=Path("run/fit.nc"), kind="hist")

    with pytest.raises(WorkflowError, match="--kind is only supported"):
        plot_command(options)


def test_plot_command_accepts_kind_for_ppc_verb() -> None:
    options = PlotOptions(verb="ppc", fit_path=Path("run/fit.nc"), kind="dist")

    command = plot_command(options)

    assert "--kind" in command.argv
    assert "dist" in command.argv


def test_idata_warmup_command_materializes_the_idata_environment_via_help() -> None:
    command = idata_warmup_command()

    assert command.argv == (
        "uvx",
        "--quiet",
        "--exclude-newer",
        BAYESITE_VIZ_EXCLUDE_NEWER,
        "--exclude-newer-package",
        f"bayesite-idata={FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION}",
        "--from",
        BAYESITE_IDATA_SOURCE,
        "bayesite-idata",
        "--help",
    )
    assert command.output_paths == ()


def test_plot_warmup_command_materializes_the_viz_environment_via_help() -> None:
    command = plot_warmup_command()

    assert command.argv == (
        "uvx",
        "--quiet",
        "--exclude-newer",
        BAYESITE_VIZ_EXCLUDE_NEWER,
        "--exclude-newer-package",
        f"bayesite-viz={FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION}",
        "--from",
        BAYESITE_VIZ_SOURCE,
        "bayesite-viz",
        "--help",
    )
    assert command.output_paths == ()


def test_idata_warmup_command_honors_source_override() -> None:
    command = idata_warmup_command(source="bayesite-idata==9.9.9")

    assert command.argv[7] == "bayesite-idata==9.9.9"


def test_plot_warmup_command_honors_source_override() -> None:
    command = plot_warmup_command(source="bayesite-viz==9.9.9")

    assert command.argv[7] == "bayesite-viz==9.9.9"


def test_warmup_commands_returns_both_entry_point_warmups_in_order() -> None:
    commands = warmup_commands()

    assert commands == (idata_warmup_command(), plot_warmup_command())
