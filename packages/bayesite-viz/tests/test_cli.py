from __future__ import annotations

from pathlib import Path

import pytest
from bayesite_viz import plots
from bayesite_viz.cli import cli
from click.testing import CliRunner


def _run(args: list[str]) -> tuple[int, str, str]:
    res = CliRunner().invoke(cli, args)
    return res.exit_code, res.stdout, res.stderr


def _read_path(stdout: str) -> Path:
    line = stdout.strip().splitlines()[-1]
    return Path(line)


def test_cli_trace_writes_png_and_prints_abs_path(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(["trace", str(continuous_fit), "-o", str(tmp_path / "t.png")])
    assert code == 0, err
    p = _read_path(out)
    assert p == (tmp_path / "t.png").resolve()
    assert p.exists() and p.stat().st_size > 0
    assert p.suffix == ".png"


def test_cli_default_filename_increments(
    continuous_fit: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    code, out, _ = _run(["trace", str(continuous_fit)])
    assert code == 0
    p = _read_path(out)
    assert p.name == "trace.png"
    # second run must not overwrite -> trace-2.png
    code, out, _ = _run(["trace", str(continuous_fit)])
    assert code == 0
    assert _read_path(out).name == "trace-2.png"


def test_cli_markdown_format(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(
        [
            "trace",
            str(continuous_fit),
            "-o",
            str(tmp_path / "t.png"),
            "-f",
            "markdown",
            "--alt",
            "hi",
        ]
    )
    assert code == 0, err
    assert out.startswith("![hi](")
    assert str((tmp_path / "t.png").resolve()) in out


def test_cli_alt_auto_generated_when_omitted(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(
        [
            "trace",
            str(continuous_fit),
            "-o",
            str(tmp_path / "t.png"),
            "-f",
            "markdown",
            "--var",
            "mu",
        ]
    )
    assert code == 0, err
    assert out.startswith("![trace plot of mu from continuous.nc](")


def test_cli_alt_format_auto(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(
        ["forest", str(continuous_fit), "-o", str(tmp_path / "f.png"), "-f", "alt"]
    )
    assert code == 0, err
    assert out.strip() == "forest plot of all variables from continuous.nc"


def test_cli_svg(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(["trace", str(continuous_fit), "-o", str(tmp_path / "t.svg"), "--svg"])
    assert code == 0, err
    assert _read_path(out).suffix == ".svg"
    assert (tmp_path / "t.svg").stat().st_size > 0


def test_cli_svg_with_png_output_path_errors(continuous_fit: Path, tmp_path: Path) -> None:
    # --svg decides the format; a .png -o contradicts it.
    code, _out, err = _run(["trace", str(continuous_fit), "-o", str(tmp_path / "t.png"), "--svg"])
    assert code != 0
    assert "--svg" in err or ".svg" in err


def test_cli_svg_output_path_without_svg_flag_errors(continuous_fit: Path, tmp_path: Path) -> None:
    # .svg extension requires --svg; PNG is the default.
    code, _out, err = _run(["trace", str(continuous_fit), "-o", str(tmp_path / "t.svg")])
    assert code != 0
    assert "--svg" in err or ".svg" in err


def test_cli_directory_arg_errors(tmp_path: Path) -> None:
    code, _out, err = _run(["trace", str(tmp_path)])
    assert code == 2
    assert "directory" in err


@pytest.mark.parametrize("verb", ["rank", "forest", "energies", "autocorr", "posterior"])
def test_cli_1to1_verbs(verb: str, continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run([verb, str(continuous_fit), "-o", str(tmp_path / f"{verb}.png")])
    assert code == 0, err
    p = _read_path(out)
    assert p.exists() and p.stat().st_size > 0


def test_cli_ess_rhat_writes_png_and_prints_abs_path(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(["ess-rhat", str(continuous_fit), "-o", str(tmp_path / "er.png")])
    assert code == 0, err
    p = _read_path(out)
    assert p == (tmp_path / "er.png").resolve()
    assert p.exists() and p.stat().st_size > 0
    assert p.suffix == ".png"


def test_cli_ess_rhat_accepts_var_selection(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(
        ["ess-rhat", str(continuous_fit), "-o", str(tmp_path / "er.png"), "--var", "mu"]
    )
    assert code == 0, err
    assert _read_path(out).exists()


def test_cli_pair_requires_var(continuous_fit: Path, tmp_path: Path) -> None:
    code, _out, err = _run(["pair", str(continuous_fit), "-o", str(tmp_path / "p.png")])
    assert code == 4
    assert "requires --var" in err


def test_cli_pair_with_var(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(
        ["pair", str(continuous_fit), "-o", str(tmp_path / "p.png"), "--var", "mu", "--var", "tau"]
    )
    assert code == 0, err
    assert _read_path(out).exists()


@pytest.mark.parametrize("kind", ["dist", "interval", "pit", "tstat"])
def test_cli_ppc_kinds(kind: str, continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(
        ["ppc", str(continuous_fit), "-o", str(tmp_path / "ppc.png"), "--kind", kind]
    )
    assert code == 0, err
    assert _read_path(out).exists()


def test_cli_ppc_rootogram_discrete(discrete_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(
        ["ppc", str(discrete_fit), "-o", str(tmp_path / "ro.png"), "--kind", "rootogram"]
    )
    assert code == 0, err
    assert _read_path(out).exists()


def test_cli_ppc_scalar_observed_data_is_controlled_error(tmp_path: Path) -> None:
    import arviz_base as azb
    import numpy as np

    rng = np.random.default_rng(12)
    dt = azb.from_dict(
        {
            "posterior": {"mu": rng.standard_normal((2, 20))},
            "posterior_predictive": {"y": rng.standard_normal((2, 20))},
            "observed_data": {"y": 0.25},
        }
    )
    fit = tmp_path / "scalar.nc"
    dt.to_netcdf(fit)

    code, _out, err = _run(["ppc", str(fit), "-o", str(tmp_path / "ppc.png")])

    assert code == 4
    assert "scalar observed variable 'y'" in err
    assert "Traceback" not in err


def test_cli_ppc_coords_scalar_slice_is_controlled_error(tmp_path: Path) -> None:
    import arviz_base as azb
    import numpy as np

    rng = np.random.default_rng(14)
    dt = azb.from_dict(
        {
            "posterior": {"mu": rng.standard_normal((2, 20))},
            "posterior_predictive": {"y": rng.standard_normal((2, 20, 2))},
            "observed_data": {"y": np.array([0.25, -0.25])},
        },
        coords={"obs": ["first", "second"]},
        dims={"y": ["obs"]},
    )
    fit = tmp_path / "coords_scalar.nc"
    dt.to_netcdf(fit)

    code, _out, err = _run(
        ["ppc", str(fit), "-o", str(tmp_path / "ppc.png"), "--coords", "obs=first"]
    )

    assert code == 4
    assert "scalar observed variable 'y'" in err
    assert "Traceback" not in err


def test_cli_ppc_negated_var_filter_still_checks_remaining_scalars(tmp_path: Path) -> None:
    import arviz_base as azb
    import numpy as np

    rng = np.random.default_rng(15)
    dt = azb.from_dict(
        {
            "posterior": {"mu": rng.standard_normal((2, 20))},
            "posterior_predictive": {
                "scalar": rng.standard_normal((2, 20)),
                "vector": rng.standard_normal((2, 20, 2)),
            },
            "observed_data": {"scalar": 0.25, "vector": np.array([0.1, 0.2])},
        },
        coords={"obs": ["first", "second"]},
        dims={"vector": ["obs"]},
    )
    fit = tmp_path / "negated_scalar.nc"
    dt.to_netcdf(fit)

    code, _out, err = _run(["ppc", str(fit), "-o", str(tmp_path / "ppc.png"), "--var", "~vector"])

    assert code == 4
    assert "scalar observed variable 'scalar'" in err
    assert "Traceback" not in err


def test_cli_ppc_mixed_negated_var_filter_checks_remaining_scalars(tmp_path: Path) -> None:
    import arviz_base as azb
    import numpy as np

    rng = np.random.default_rng(16)
    dt = azb.from_dict(
        {
            "posterior": {"mu": rng.standard_normal((2, 20))},
            "posterior_predictive": {
                "ignore": rng.standard_normal((2, 20, 2)),
                "scalar": rng.standard_normal((2, 20)),
                "vector": rng.standard_normal((2, 20, 2)),
            },
            "observed_data": {
                "ignore": np.array([-0.1, -0.2]),
                "scalar": 0.25,
                "vector": np.array([0.1, 0.2]),
            },
        },
        coords={"obs": ["first", "second"]},
        dims={"ignore": ["obs"], "vector": ["obs"]},
    )
    fit = tmp_path / "mixed_negated_scalar.nc"
    dt.to_netcdf(fit)

    code, _out, err = _run(
        [
            "ppc",
            str(fit),
            "-o",
            str(tmp_path / "ppc.png"),
            "--var",
            "vector",
            "--var",
            "~ignore",
        ]
    )

    assert code == 4
    assert "scalar observed variable 'scalar'" in err
    assert "Traceback" not in err


def test_cli_ppc_literal_tilde_var_name_is_not_treated_as_negation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import arviz_base as azb
    import bayesite_viz.cli as cli_module
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(17)
    dt = azb.from_dict(
        {
            "posterior": {"mu": rng.standard_normal((2, 20))},
            "posterior_predictive": {
                "~y": rng.standard_normal((2, 20, 2)),
                "scalar": rng.standard_normal((2, 20)),
            },
            "observed_data": {"~y": np.array([0.1, 0.2]), "scalar": 0.25},
        },
        coords={"obs": ["first", "second"]},
        dims={"~y": ["obs"]},
    )

    def _fake_load_fit(_fit: Path) -> xr.DataTree:
        return dt

    monkeypatch.setattr(cli_module, "load_fit", _fake_load_fit)

    code, out, err = _run(
        ["ppc", "literal_tilde.nc", "-o", str(tmp_path / "tilde.png"), "--var", "~y"]
    )

    assert code == 0, err
    assert _read_path(out).exists()


def test_cli_ppc_tstat_allows_scalar_observed_data(tmp_path: Path) -> None:
    import arviz_base as azb
    import numpy as np

    rng = np.random.default_rng(13)
    dt = azb.from_dict(
        {
            "posterior": {"mu": rng.standard_normal((2, 20))},
            "posterior_predictive": {"y": rng.standard_normal((2, 20))},
            "observed_data": {"y": 0.25},
        }
    )
    fit = tmp_path / "scalar_tstat.nc"
    dt.to_netcdf(fit)

    code, out, err = _run(["ppc", str(fit), "-o", str(tmp_path / "tstat.png"), "--kind", "tstat"])

    assert code == 0, err
    assert _read_path(out).exists()


def test_cli_ppc_missing_groups_errors(tmp_path: Path, continuous_fit: Path) -> None:
    # continuous_fit has ppc groups, so build a fit without them
    import arviz_base as azb
    import numpy as np

    rng = np.random.default_rng(2)
    dt = azb.from_dict({"posterior": {"mu": rng.standard_normal((4, 100))}})
    fit = tmp_path / "post.nc"
    dt.to_netcdf(fit)
    code, _out, err = _run(["ppc", str(fit), "-o", str(tmp_path / "x.png")])
    assert code == 3
    assert "posterior_predictive" in err


def _fit_without(tmp_path: Path, *, drop: str) -> Path:
    """Build a fit that lacks the given group."""
    import arviz_base as azb
    import numpy as np

    rng = np.random.default_rng(3)
    groups = {
        "posterior": {"mu": rng.standard_normal((4, 100))},
        "sample_stats": {"energy": rng.standard_normal((4, 100))},
    }
    if drop in groups:
        del groups[drop]
    dt = azb.from_dict(groups)
    fit = tmp_path / f"no_{drop}.nc"
    dt.to_netcdf(fit)
    return fit


@pytest.mark.parametrize("verb", ["trace", "rank", "forest", "posterior", "autocorr", "ess-rhat"])
def test_cli_posterior_verbs_require_posterior_group(verb: str, tmp_path: Path) -> None:
    fit = _fit_without(tmp_path, drop="posterior")
    code, _out, err = _run([verb, str(fit), "-o", str(tmp_path / "x.png")])
    assert code == 3
    assert "posterior" in err


def test_cli_energies_requires_sample_stats(tmp_path: Path) -> None:
    fit = _fit_without(tmp_path, drop="sample_stats")
    code, _out, err = _run(["energies", str(fit), "-o", str(tmp_path / "e.png")])
    assert code == 3
    assert "sample_stats" in err


def test_cli_energies_ignores_var_and_coords(continuous_fit: Path, tmp_path: Path) -> None:
    # plot_energy does not accept var_names/coords as selectors; the shared flags
    # must not be forwarded into it as unmapped aesthetics.
    code, out, err = _run(
        ["energies", str(continuous_fit), "-o", str(tmp_path / "e.png"), "--var", "mu"]
    )
    assert code == 0, err
    assert _read_path(out).exists()


def test_cli_energies_requires_energy_variable(tmp_path: Path) -> None:
    # sample_stats present but without an `energy` variable (non-HMC fit):
    # must be a repair message, not an AttributeError traceback.
    import arviz_base as azb
    import numpy as np

    rng = np.random.default_rng(4)
    dt = azb.from_dict(
        {
            "posterior": {"mu": rng.standard_normal((4, 100))},
            "sample_stats": {"diverging": rng.random((4, 100)) < 0.1},
        }
    )
    fit = tmp_path / "no_energy.nc"
    dt.to_netcdf(fit)
    code, _out, err = _run(["energies", str(fit), "-o", str(tmp_path / "e.png")])
    assert code == 3
    assert "energy" in err
    assert "Traceback" not in err


def test_cli_pair_requires_posterior_group(tmp_path: Path) -> None:
    fit = _fit_without(tmp_path, drop="posterior")
    code, _out, err = _run(["pair", str(fit), "-o", str(tmp_path / "p.png"), "--var", "mu"])
    assert code == 3
    assert "posterior" in err


def test_cli_posterior_kind(continuous_fit: Path, tmp_path: Path) -> None:
    for kind in plots.DIST_KINDS:
        code, out, err = _run(
            ["posterior", str(continuous_fit), "-o", str(tmp_path / f"{kind}.png"), "--kind", kind]
        )
        assert code == 0, err
        assert _read_path(out).exists()


def test_cli_posterior_kind_rejects_unknown(continuous_fit: Path, tmp_path: Path) -> None:
    code, _out, err = _run(
        ["posterior", str(continuous_fit), "-o", str(tmp_path / "x.png"), "--kind", "bogus"]
    )
    assert code != 0


def test_cli_var_and_coords(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(
        ["trace", str(continuous_fit), "-o", str(tmp_path / "v.png"), "--var", "mu"]
    )
    assert code == 0, err
    assert _read_path(out).exists()


def test_cli_missing_var_is_controlled_error(continuous_fit: Path, tmp_path: Path) -> None:
    # a --var that doesn't exist in the fit makes arviz raise KeyError; that must
    # be a controlled exit, not a traceback.
    code, _out, err = _run(
        ["trace", str(continuous_fit), "-o", str(tmp_path / "x.png"), "--var", "nonexistent"]
    )
    assert code == 4
    assert "Traceback" not in err


def test_cli_bad_coords_is_controlled_error(continuous_fit: Path, tmp_path: Path) -> None:
    code, _out, err = _run(
        [
            "trace",
            str(continuous_fit),
            "-o",
            str(tmp_path / "x.png"),
            "--coords",
            "school=NoSuchSchool",
        ]
    )
    assert code != 0
    assert "Traceback" not in err


def test_cli_regex_flag_selects_variables(continuous_fit: Path, tmp_path: Path) -> None:
    # invariants.md documents --regex / --like as the variable-selection flags
    code, out, err = _run(
        ["trace", str(continuous_fit), "-o", str(tmp_path / "r.png"), "--regex", "--var", "^m"]
    )
    assert code == 0, err
    assert _read_path(out).exists()


def test_cli_like_flag_selects_variables(continuous_fit: Path, tmp_path: Path) -> None:
    code, out, err = _run(
        ["trace", str(continuous_fit), "-o", str(tmp_path / "l.png"), "--like", "--var", "mu"]
    )
    assert code == 0, err
    assert _read_path(out).exists()


def test_cli_raw_ndjson_repair_message(tmp_path: Path) -> None:
    p = tmp_path / "posterior.ndjson"
    p.write_text('{"mu": 0.1}\n', encoding="utf-8")
    code, _out, err = _run(["trace", str(p), "-o", str(tmp_path / "x.png")])
    assert code == 5
    assert "bayesite-idata" in err


def test_cli_unsupported_backend_clean_error(continuous_fit: Path, tmp_path: Path) -> None:
    # bokeh is not installed in v1; selecting it must be a controlled error,
    # not a Python traceback.
    code, _out, err = _run(
        ["trace", str(continuous_fit), "-o", str(tmp_path / "b.png"), "--backend", "bokeh"]
    )
    assert code == 6
    assert "bokeh" in err.lower()
    assert "install" in err.lower()
    assert "Traceback" not in err


def test_run_backend_export_runtime_error_is_controlled(
    continuous_fit: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A backend present but missing its export dep (bokeh/selenium -> RuntimeError,
    # plotly/kaleido -> ValueError) must surface as exit 6, not a traceback.
    from arviz_plots import PlotCollection

    def _bad_savefig(self: PlotCollection, filename: object, **kwargs: object) -> None:
        msg = "selenium not installed"
        raise RuntimeError(msg)

    monkeypatch.setattr(PlotCollection, "savefig", _bad_savefig)
    code, _out, err = _run(["trace", str(continuous_fit), "-o", str(tmp_path / "x.png")])
    assert code == 6
    assert "cannot save" in err
    assert "Traceback" not in err
