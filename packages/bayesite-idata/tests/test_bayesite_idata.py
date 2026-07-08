from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import cast

import arviz_base as azb
import pytest
import xarray as xr
from bayesite_idata import from_run_dir, write_netcdf
from bayesite_idata import validate as validate_module
from bayesite_idata.cli import cli as idata_cli
from bayesite_idata.protocol_v0 import (
    ProtocolError,
    read_posterior_predictive_stream,
    read_posterior_stream,
    read_prior_predictive_stream,
)
from bayesite_idata.run_dir import RunDirError, discover_run_dir
from click.testing import CliRunner


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _posterior_lines(chains: tuple[int, ...] = (0, 1)) -> list[dict[str, object]]:
    draw_count = len(chains) * 2
    header: dict[str, object] = {
        "draws_format": "v0-provisional",
        "artifact_kind": "posterior_draws",
        "artifact_scope": "observed_data_conditioned_parameter_draws",
        "posterior_identity_hash": "abc123",
        "params": [
            {"name": "mu", "shape": []},
            {"name": "beta", "shape": [2]},
        ],
        "parameter_order": ["mu", "beta"],
        "settings": {"num_warmup": 1, "num_draws": 2},
        "seed": 11,
        "chain_count": len(chains),
        "chain_order": list(chains),
        "draw_count": draw_count,
        "chains": len(chains),
        "sample_stats_mode": "per_draw_v1",
    }
    draws: list[dict[str, object]] = []
    for chain in chains:
        for draw in (0, 1):
            base = chain * 10 + draw
            draws.append(
                {
                    "draws_format": "v0-provisional",
                    "artifact_kind": "posterior_draws",
                    "artifact_scope": "observed_data_conditioned_parameter_draws",
                    "draw_index": len(draws),
                    "seed": 11,
                    "draw_count": draw_count,
                    "chain_count": len(chains),
                    "chain_order": list(chains),
                    "chain": chain,
                    "draw": draw,
                    "parameter_order": ["mu", "beta"],
                    "values": {"mu": float(base), "beta": [float(base + 1), float(base + 2)]},
                    "sample_stats_mode": "per_draw_v1",
                    "diverging": chain == chains[-1] and draw == 0,
                    "tree_depth": 3 + draw,
                    "tree_accept": 0.8 + 0.01 * base,
                }
            )
    trailer_chains = [
        {
            "chain": chain,
            "draw_count": 2,
            "divergences": 1 if chain == chains[-1] else 0,
            "step_size": 0.1 + 0.01 * chain,
            "mean_accept": 0.8,
        }
        for chain in chains
    ]
    trailer: dict[str, object] = {
        "trailer": {
            "draws_format": "v0-provisional",
            "artifact_kind": "posterior_draws",
            "artifact_scope": "observed_data_conditioned_parameter_draws",
            "posterior_identity_hash": "abc123",
            "seed": 11,
            "draws_per_chain": 2,
            "chain_count": len(chains),
            "chain_order": list(chains),
            "draw_count": draw_count,
            "parameter_order": ["mu", "beta"],
            "chains": trailer_chains,
            "rhat": {"mu": None, "beta": None},
            "ess": {"mu": None, "beta": None},
        }
    }
    return [header, *draws, trailer]


def _prior_predictive_lines(draws: int = 3) -> list[dict[str, object]]:
    sites: list[dict[str, object]] = [
        {
            "name": "y",
            "stochastic_site": "y",
            "role": "observed",
            "shape": [2],
            "integer": True,
        }
    ]
    header: dict[str, object] = {
        "prior_predictive_format": "v0-provisional",
        "artifact_kind": "prior_predictive_draws",
        "artifact_scope": "declared_data_conditioned_site_draws",
        "seed": 10,
        "draw_count": draws,
        "draws": draws,
        "site_count": 1,
        "site_order": ["y"],
        "sites": sites,
    }
    records: list[dict[str, object]] = [
        {
            "prior_predictive_format": "v0-provisional",
            "artifact_kind": "prior_predictive_draws",
            "artifact_scope": "declared_data_conditioned_site_draws",
            "draw_index": draw,
            "draw": draw,
            "draw_count": draws,
            "site_count": 1,
            "site_order": ["y"],
            "values": {"y": [draw, draw + 1]},
        }
        for draw in range(draws)
    ]
    trailer: dict[str, object] = {
        "trailer": {
            "prior_predictive_format": "v0-provisional",
            "artifact_kind": "prior_predictive_draws",
            "artifact_scope": "declared_data_conditioned_site_draws",
            "seed": 10,
            "draw_count": draws,
            "draws": draws,
            "site_count": 1,
            "site_order": ["y"],
        }
    }
    return [header, *records, trailer]


def _posterior_predictive_lines(chains: tuple[int, ...] = (0, 1)) -> list[dict[str, object]]:
    sites: list[dict[str, object]] = [
        {
            "name": "y",
            "stochastic_site": "y",
            "role": "observed",
            "shape": [2],
            "integer": True,
        }
    ]
    header: dict[str, object] = {
        "posterior_predictive_format": "v0-provisional",
        "artifact_kind": "posterior_predictive_draws",
        "artifact_scope": "observed_data_conditioned_replicated_observed_data_draws",
        "seed": 12,
        "source_fit_seed": 11,
        "draw_count": len(chains) * 2,
        "site_count": 1,
        "site_order": ["y"],
        "sites": sites,
    }
    draws: list[dict[str, object]] = []
    for chain in chains:
        for draw in (0, 1):
            draws.append(
                {
                    "posterior_predictive_format": "v0-provisional",
                    "artifact_kind": "posterior_predictive_draws",
                    "artifact_scope": "observed_data_conditioned_replicated_observed_data_draws",
                    "draw_index": len(draws),
                    "seed": 12,
                    "draw_count": len(chains) * 2,
                    "source_fit_draw_index": len(draws),
                    "source_chain": chain,
                    "source_draw": draw,
                    "site_count": 1,
                    "site_order": ["y"],
                    "values": {"y": [chain + draw, chain + draw + 1]},
                }
            )
    trailer: dict[str, object] = {
        "trailer": {
            "posterior_predictive_format": "v0-provisional",
            "artifact_kind": "posterior_predictive_draws",
            "artifact_scope": "observed_data_conditioned_replicated_observed_data_draws",
            "seed": 12,
            "source_fit_seed": 11,
            "draw_count": len(chains) * 2,
            "site_count": 1,
            "site_order": ["y"],
        }
    }
    return [header, *draws, trailer]


def _write_ndjson(path: Path, lines: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _write_fake_bayesite(path: Path, log: Path, *, exit_code: int = 0, stderr: str = "") -> None:
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(log))}\n"
        f"printf '%s\\n' {shlex.quote(stderr)} >&2\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.fixture
def bayesite_run(tmp_path: Path) -> Path:
    _write_json(tmp_path / "model.ir.json", {"bayeswire_ir": 1, "model": {"node": "ModelMeta"}})
    _write_json(
        tmp_path / "data.json",
        {
            "obs": {"dtype": "float64", "shape": [2], "values": [1.0, 2.0]},
            "n": {"dtype": "int32", "shape": [], "values": [2]},
            "flag": {"dtype": "bool", "shape": [2], "values": [True, False]},
            "y": {"dtype": "int64", "shape": [2], "values": [3, 4]},
        },
    )
    _write_ndjson(tmp_path / "posterior.ndjson", _posterior_lines())
    return tmp_path


def test_discover_run_dir_requires_core_artifacts(tmp_path: Path) -> None:
    with pytest.raises(RunDirError) as exc:
        discover_run_dir(tmp_path)
    assert "model.ir.json" in str(exc.value)


def test_discover_run_dir_finds_optional_artifacts(bayesite_run: Path) -> None:
    (bayesite_run / "posterior_predictive.ndjson").write_text("", encoding="utf-8")
    run = discover_run_dir(bayesite_run)
    assert run.posterior == bayesite_run / "posterior.ndjson"
    assert run.posterior_predictive == bayesite_run / "posterior_predictive.ndjson"
    assert run.prior_predictive is None


def test_discover_run_dir_rejects_optional_artifact_directories(bayesite_run: Path) -> None:
    (bayesite_run / "prior_predictive.ndjson").mkdir()
    with pytest.raises(RunDirError, match="not a file"):
        discover_run_dir(bayesite_run)


def test_read_posterior_stream_checks_shapes_and_sample_stats(bayesite_run: Path) -> None:
    stream = read_posterior_stream(bayesite_run / "posterior.ndjson")
    assert [p.name for p in stream.params] == ["mu", "beta"]
    assert stream.chains == (0, 1)
    assert stream.draws_per_chain == 2
    assert stream.records[2].diverging is True


def test_read_posterior_stream_accepts_per_draw_v2_energy(bayesite_run: Path) -> None:
    lines = _posterior_lines()
    lines[0]["sample_stats_mode"] = "per_draw_v2"
    for index, line in enumerate(lines[1:-1]):
        line["sample_stats_mode"] = "per_draw_v2"
        line["energy"] = 100.0 + index
    _write_ndjson(bayesite_run / "posterior.ndjson", lines)

    stream = read_posterior_stream(bayesite_run / "posterior.ndjson")

    assert stream.sample_stats_mode == "per_draw_v2"
    assert stream.records[0].energy == 100.0


def test_read_posterior_stream_rejects_unknown_format(bayesite_run: Path) -> None:
    lines = _posterior_lines()
    lines[0]["draws_format"] = "vNEXT"
    bad = bayesite_run / "bad.ndjson"
    _write_ndjson(bad, lines)
    with pytest.raises(ProtocolError, match="draws_format"):
        read_posterior_stream(bad)


def test_read_posterior_stream_rejects_duplicate_chain_draw(bayesite_run: Path) -> None:
    lines = _posterior_lines()
    lines[2]["draw"] = 0
    bad = bayesite_run / "dup.ndjson"
    _write_ndjson(bad, lines)
    with pytest.raises(ProtocolError, match="duplicate"):
        read_posterior_stream(bad)


def test_read_posterior_stream_rejects_duplicate_chain_order_labels(bayesite_run: Path) -> None:
    lines = _posterior_lines()
    lines[0]["chain_order"] = [0, 0]
    bad = bayesite_run / "dup_chain_order.ndjson"
    _write_ndjson(bad, lines)
    with pytest.raises(ProtocolError, match="duplicate chain_order"):
        read_posterior_stream(bad)


def test_read_posterior_stream_rejects_value_shape_mismatch(bayesite_run: Path) -> None:
    lines = _posterior_lines()
    values = lines[1]["values"]
    assert isinstance(values, dict)
    cast(dict[str, object], values)["beta"] = [1.0]
    bad = bayesite_run / "shape.ndjson"
    _write_ndjson(bad, lines)
    with pytest.raises(ProtocolError, match="shape"):
        read_posterior_stream(bad)


def test_read_posterior_stream_rejects_truncated_records(bayesite_run: Path) -> None:
    lines = _posterior_lines()
    # Remove draw=1 from both chains. A lenient reader must not infer a smaller
    # fit when header/trailer counts still announce four retained draws.
    truncated = [lines[0], lines[1], lines[3], lines[-1]]
    bad = bayesite_run / "truncated.ndjson"
    _write_ndjson(bad, truncated)
    with pytest.raises(ProtocolError, match="draw_count"):
        read_posterior_stream(bad)


def test_read_prior_predictive_stream_rejects_truncated_records(bayesite_run: Path) -> None:
    lines = _prior_predictive_lines(draws=3)
    truncated = [lines[0], lines[1], lines[-1]]
    bad = bayesite_run / "truncated_prior.ndjson"
    _write_ndjson(bad, truncated)
    with pytest.raises(ProtocolError, match="draw_count"):
        read_prior_predictive_stream(bad)


def test_read_posterior_predictive_stream_rejects_truncated_records(bayesite_run: Path) -> None:
    lines = _posterior_predictive_lines(chains=(0, 1))
    truncated = [lines[0], lines[1], lines[-1]]
    bad = bayesite_run / "truncated_pp.ndjson"
    _write_ndjson(bad, truncated)
    with pytest.raises(ProtocolError, match="draw_count"):
        read_posterior_predictive_stream(bad)


def test_from_run_dir_wraps_malformed_data_json(bayesite_run: Path) -> None:
    (bayesite_run / "data.json").write_text('{"obs": ', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid data.json"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_rejects_malformed_typed_data_values(bayesite_run: Path) -> None:
    _write_json(
        bayesite_run / "data.json",
        {
            "count": {"dtype": "int64", "shape": [1], "values": [1.9]},
            "flag": {"dtype": "bool", "shape": [1], "values": ["False"]},
        },
    )
    with pytest.raises(ValueError, match="integer dtype"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_rejects_empty_optional_predictive_artifact(bayesite_run: Path) -> None:
    (bayesite_run / "posterior_predictive.ndjson").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="expected header"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_builds_datatree_with_auto_dims_and_stats(bayesite_run: Path) -> None:
    dt = from_run_dir(bayesite_run, validate="skip")
    assert isinstance(dt, xr.DataTree)
    posterior = dt["/posterior"].ds
    assert posterior["mu"].dims == ("chain", "draw")
    assert posterior["beta"].dims == ("chain", "draw", "beta_dim_0")
    assert posterior["beta"].shape == (2, 2, 2)
    assert posterior["beta_dim_0"].values.tolist() == [0, 1]
    stats = dt["/sample_stats"].ds
    assert "acceptance_rate" in stats
    assert stats["diverging"].values.tolist() == [[False, False], [True, False]]
    observed = dt["/observed_data"].ds
    assert observed["obs"].dims == ("obs_dim_0",)
    assert observed["n"].shape == ()
    assert observed["n"].dtype.kind in {"i", "u"}
    assert observed["flag"].dtype.kind == "b"


def test_from_run_dir_exports_per_draw_v2_energy(bayesite_run: Path) -> None:
    lines = _posterior_lines()
    lines[0]["sample_stats_mode"] = "per_draw_v2"
    for index, line in enumerate(lines[1:-1]):
        line["sample_stats_mode"] = "per_draw_v2"
        line["energy"] = 100.0 + index
    _write_ndjson(bayesite_run / "posterior.ndjson", lines)

    dt = from_run_dir(bayesite_run, validate="skip")

    stats = dt["/sample_stats"].ds
    assert stats["energy"].dims == ("chain", "draw")
    assert stats["energy"].values.tolist() == [[100.0, 101.0], [102.0, 103.0]]


def test_from_run_dir_pins_sample_dim_order(
    bayesite_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(azb.rcParams, "data.sample_dims", ("draw", "chain"))
    dt = from_run_dir(bayesite_run, validate="skip")
    assert dt["/posterior"].ds["beta"].dims[:2] == ("chain", "draw")


def test_from_run_dir_applies_dims_json_to_named_variables(bayesite_run: Path) -> None:
    _write_json(
        bayesite_run / "data.json",
        {
            "x": {"dtype": "float64", "shape": [2, 2], "values": [1.0, 2.0, 3.0, 4.0]},
            "y": {"dtype": "int64", "shape": [2], "values": [3, 4]},
        },
    )
    _write_ndjson(bayesite_run / "prior_predictive.ndjson", _prior_predictive_lines(draws=3))
    _write_ndjson(
        bayesite_run / "posterior_predictive.ndjson", _posterior_predictive_lines(chains=(0, 1))
    )
    _write_json(
        bayesite_run / "dims.json",
        {
            "dims_format": "bayescycle-dims-v1",
            "dims": {
                "mu": [],
                "beta": ["predictor"],
                "x": ["obs", "predictor"],
                "y": ["obs"],
            },
            "coords": {"predictor": ["x1", "x2"], "obs": ["row0", "row1"]},
        },
    )

    dt = from_run_dir(bayesite_run, validate="skip")

    posterior = dt["/posterior"].ds
    assert posterior["mu"].dims == ("chain", "draw")
    assert posterior["beta"].dims == ("chain", "draw", "predictor")
    assert posterior["predictor"].values.tolist() == ["x1", "x2"]

    observed = dt["/observed_data"].ds
    assert observed["y"].dims == ("obs",)
    assert observed["obs"].values.tolist() == ["row0", "row1"]

    constants = dt["/constant_data"].ds
    assert constants["x"].dims == ("obs", "predictor")
    assert constants["predictor"].values.tolist() == ["x1", "x2"]

    prior_pred = dt["/prior_predictive"].ds
    assert prior_pred["y"].dims == ("chain", "draw", "obs")

    posterior_pred = dt["/posterior_predictive"].ds
    assert posterior_pred["y"].dims == ("chain", "draw", "obs")


def test_from_run_dir_keeps_prior_and_prior_predictive_coords_separate(
    bayesite_run: Path,
) -> None:
    _write_json(
        bayesite_run / "data.json",
        {"y": {"dtype": "int64", "shape": [2], "values": [3, 4]}},
    )
    lines = _prior_predictive_lines(draws=3)
    header = lines[0]
    raw_sites = cast(list[object], header["sites"])
    raw_sites.insert(
        0,
        {
            "name": "theta",
            "stochastic_site": "theta",
            "role": "parameter",
            "shape": [2],
            "integer": False,
        },
    )
    header["site_count"] = 2
    header["site_order"] = ["theta", "y"]
    for draw, line in enumerate(lines[1:-1]):
        line["site_count"] = 2
        line["site_order"] = ["theta", "y"]
        raw_values = cast(dict[str, object], line["values"])
        raw_values["theta"] = [float(draw), float(draw + 1)]
    raw_trailer = cast(dict[str, object], lines[-1]["trailer"])
    raw_trailer["site_count"] = 2
    raw_trailer["site_order"] = ["theta", "y"]
    _write_ndjson(bayesite_run / "prior_predictive.ndjson", lines)
    _write_json(
        bayesite_run / "dims.json",
        {
            "dims_format": "bayescycle-dims-v1",
            "dims": {"theta": ["y"], "y": ["obs"]},
            "coords": {"y": ["left", "right"], "obs": [0, 1]},
        },
    )

    dt = from_run_dir(bayesite_run, validate="skip")

    prior = dt["/prior"].ds
    assert prior["theta"].dims == ("chain", "draw", "y")
    assert prior["y"].values.tolist() == ["left", "right"]
    prior_pred = dt["/prior_predictive"].ds
    assert prior_pred["y"].dims == ("chain", "draw", "obs")
    assert prior_pred["obs"].values.tolist() == [0, 1]


def test_from_run_dir_rejects_dims_json_rank_mismatch(bayesite_run: Path) -> None:
    _write_json(
        bayesite_run / "dims.json",
        {"dims_format": "bayescycle-dims-v1", "dims": {"beta": ["row", "col"]}},
    )

    with pytest.raises(ValueError, match="rank"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_rejects_dims_json_coord_length_mismatch(bayesite_run: Path) -> None:
    _write_json(
        bayesite_run / "dims.json",
        {
            "dims_format": "bayescycle-dims-v1",
            "dims": {"beta": ["predictor"]},
            "coords": {"predictor": ["x1"]},
        },
    )

    with pytest.raises(ValueError, match="coord 'predictor' length"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_rejects_dims_json_reserved_sample_dim(bayesite_run: Path) -> None:
    _write_json(
        bayesite_run / "dims.json",
        {"dims_format": "bayescycle-dims-v1", "dims": {"beta": ["chain"]}},
    )

    with pytest.raises(ValueError, match="reserved sample dimension"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_rejects_dims_json_variable_coord_conflict(bayesite_run: Path) -> None:
    _write_json(
        bayesite_run / "dims.json",
        {"dims_format": "bayescycle-dims-v1", "dims": {"beta": ["beta"]}},
    )

    with pytest.raises(ValueError, match="coordinate name.*variable name"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_rejects_dims_json_unknown_format(bayesite_run: Path) -> None:
    _write_json(bayesite_run / "dims.json", {"dims_format": "vNEXT", "dims": {"beta": []}})

    with pytest.raises(ValueError, match="dims_format"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_preserves_source_chain_labels(tmp_path: Path) -> None:
    _write_json(tmp_path / "model.ir.json", {"bayeswire_ir": 1, "model": {"node": "ModelMeta"}})
    _write_json(tmp_path / "data.json", {})
    _write_ndjson(tmp_path / "posterior.ndjson", _posterior_lines(chains=(2, 4)))
    dt = from_run_dir(tmp_path, validate="skip")
    assert dt["/posterior"].ds["chain"].values.tolist() == [2, 4]


def test_from_run_dir_preserves_integer_posterior_predictive_dtype(bayesite_run: Path) -> None:
    _write_ndjson(
        bayesite_run / "posterior_predictive.ndjson", _posterior_predictive_lines(chains=(0, 1))
    )
    dt = from_run_dir(bayesite_run, validate="skip")
    pp = dt["/posterior_predictive"].ds["y"]
    assert pp.dtype.kind in {"i", "u"}
    assert pp.shape == (2, 2, 2)


def test_from_run_dir_rejects_missing_observed_data_for_posterior_predictive(
    bayesite_run: Path,
) -> None:
    _write_json(
        bayesite_run / "data.json", {"x": {"dtype": "float64", "shape": [2], "values": [1.0, 2.0]}}
    )
    _write_ndjson(
        bayesite_run / "posterior_predictive.ndjson", _posterior_predictive_lines(chains=(0, 1))
    )
    with pytest.raises(ValueError, match="missing observed data"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_rejects_posterior_predictive_from_other_fit(bayesite_run: Path) -> None:
    lines = _posterior_predictive_lines(chains=(0, 1))
    header = lines[0]
    trailer = lines[-1]["trailer"]
    assert isinstance(header, dict)
    assert isinstance(trailer, dict)
    header["source_fit_seed"] = 99
    cast(dict[str, object], trailer)["source_fit_seed"] = 99
    _write_ndjson(bayesite_run / "posterior_predictive.ndjson", lines)
    with pytest.raises(ValueError, match="source_fit_seed"):
        from_run_dir(bayesite_run, validate="skip")


def test_from_run_dir_preserves_large_integer_predictive_values(bayesite_run: Path) -> None:
    lines = _posterior_predictive_lines(chains=(0, 1))
    values = lines[1]["values"]
    assert isinstance(values, dict)
    large = 9_007_199_254_740_993
    cast(dict[str, object], values)["y"] = [large, large + 2]
    _write_ndjson(bayesite_run / "posterior_predictive.ndjson", lines)
    dt = from_run_dir(bayesite_run, validate="skip")
    pp = dt["/posterior_predictive"].ds["y"]
    assert pp.values[0, 0, 0].item() == large
    assert pp.values[0, 0, 1].item() == large + 2


def test_from_run_dir_prior_predictive_uses_its_own_sample_coords(bayesite_run: Path) -> None:
    _write_ndjson(bayesite_run / "prior_predictive.ndjson", _prior_predictive_lines(draws=3))
    dt = from_run_dir(bayesite_run, validate="skip")
    assert dt["/posterior"].ds["chain"].values.tolist() == [0, 1]
    prior_pred = dt["/prior_predictive"].ds
    assert prior_pred["chain"].values.tolist() == [0]
    assert prior_pred["draw"].values.tolist() == [0, 1, 2]
    assert prior_pred["y"].dtype.kind in {"i", "u"}


def test_from_run_dir_splits_constants_from_observed_for_ppc(bayesite_run: Path) -> None:
    _write_json(
        bayesite_run / "data.json",
        {
            "x": {"dtype": "float64", "shape": [2], "values": [1.0, 2.0]},
            "y": {"dtype": "int64", "shape": [2], "values": [3, 4]},
        },
    )
    _write_ndjson(
        bayesite_run / "posterior_predictive.ndjson", _posterior_predictive_lines(chains=(0, 1))
    )
    dt = from_run_dir(bayesite_run, validate="skip")
    assert set(dt["/observed_data"].ds.data_vars) == {"y"}
    assert set(dt["/constant_data"].ds.data_vars) == {"x"}


def test_write_netcdf_round_trips(bayesite_run: Path, tmp_path: Path) -> None:
    out = tmp_path / "fit.nc"
    path = write_netcdf(from_run_dir(bayesite_run, validate="skip"), out)
    assert path == out.resolve()
    reopened = xr.open_datatree(path)
    assert "posterior" in {group.lstrip("/") for group in reopened.groups}


def test_idata_cli_writes_netcdf_and_prints_abs_path_only(
    bayesite_run: Path, tmp_path: Path
) -> None:
    out = tmp_path / "fit.nc"
    res = CliRunner().invoke(idata_cli, [str(bayesite_run), "-o", str(out), "--validate", "skip"])
    assert res.exit_code == 0, res.stderr
    assert res.stdout == str(out.resolve()) + "\n"
    assert out.exists()


def test_idata_cli_default_warn_does_not_require_bayesite_binary(
    bayesite_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate_module.shutil, "which", lambda _name: None)
    out = tmp_path / "fit.nc"
    res = CliRunner().invoke(idata_cli, [str(bayesite_run), "-o", str(out)])
    assert res.exit_code == 0, res.stderr
    assert res.stdout == str(out.resolve()) + "\n"
    assert "warning:" in res.stderr


def test_idata_cli_validate_require_uses_explicit_bayesite_binary(
    bayesite_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate_module.shutil, "which", lambda _name: None)
    exe = tmp_path / "custom-bayesite"
    log = tmp_path / "diagnose-args.txt"
    _write_fake_bayesite(exe, log)
    out = tmp_path / "fit.nc"

    res = CliRunner().invoke(
        idata_cli,
        [
            str(bayesite_run),
            "-o",
            str(out),
            "--validate",
            "require",
            "--bayesite",
            str(exe),
        ],
    )

    assert res.exit_code == 0, res.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        "diagnose",
        "--fit",
        str(bayesite_run / "posterior.ndjson"),
    ]


def test_idata_cli_validate_require_uses_relative_explicit_bayesite_binary(
    bayesite_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "relative-bayesite"
    log = tmp_path / "relative-args.txt"
    _write_fake_bayesite(exe, log)
    out = tmp_path / "fit.nc"
    monkeypatch.chdir(tmp_path)

    res = CliRunner().invoke(
        idata_cli,
        [
            str(bayesite_run),
            "-o",
            str(out),
            "--validate",
            "require",
            "--bayesite",
            exe.name,
        ],
        env={"PATH": "/nonexistent"},
    )

    assert res.exit_code == 0, res.stderr
    assert log.read_text(encoding="utf-8").splitlines()[0] == "diagnose"


def test_idata_cli_validate_require_uses_bayesite_env(
    bayesite_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate_module.shutil, "which", lambda _name: None)
    exe = tmp_path / "env-bayesite"
    log = tmp_path / "env-args.txt"
    _write_fake_bayesite(exe, log)
    out = tmp_path / "fit.nc"

    res = CliRunner().invoke(
        idata_cli,
        [str(bayesite_run), "-o", str(out), "--validate", "require"],
        env={"BAYESITE": str(exe)},
    )

    assert res.exit_code == 0, res.stderr
    assert log.read_text(encoding="utf-8").splitlines()[0] == "diagnose"


def test_idata_cli_explicit_bayesite_overrides_env(bayesite_run: Path, tmp_path: Path) -> None:
    env_exe = tmp_path / "env-bayesite"
    env_log = tmp_path / "env-args.txt"
    _write_fake_bayesite(env_exe, env_log, exit_code=19, stderr="wrong binary")
    cli_exe = tmp_path / "cli-bayesite"
    cli_log = tmp_path / "cli-args.txt"
    _write_fake_bayesite(cli_exe, cli_log)
    out = tmp_path / "fit.nc"

    res = CliRunner().invoke(
        idata_cli,
        [
            str(bayesite_run),
            "-o",
            str(out),
            "--validate",
            "require",
            "--bayesite",
            str(cli_exe),
        ],
        env={"BAYESITE": str(env_exe)},
    )

    assert res.exit_code == 0, res.stderr
    assert cli_log.exists()
    assert not env_log.exists()


def test_idata_cli_validate_require_keeps_path_lookup(
    bayesite_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BAYESITE", raising=False)
    exe = tmp_path / "path-bayesite"
    log = tmp_path / "path-args.txt"
    _write_fake_bayesite(exe, log)
    monkeypatch.setattr(validate_module.shutil, "which", lambda _name: str(exe))
    out = tmp_path / "fit.nc"

    res = CliRunner().invoke(
        idata_cli, [str(bayesite_run), "-o", str(out), "--validate", "require"]
    )

    assert res.exit_code == 0, res.stderr
    assert log.read_text(encoding="utf-8").splitlines()[0] == "diagnose"


def test_idata_cli_validate_warn_with_missing_explicit_bayesite_continues(
    bayesite_run: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-bayesite"
    out = tmp_path / "fit.nc"

    res = CliRunner().invoke(
        idata_cli,
        [
            str(bayesite_run),
            "-o",
            str(out),
            "--validate",
            "warn",
            "--bayesite",
            str(missing),
        ],
    )

    assert res.exit_code == 0, res.stderr
    assert out.exists()
    assert "warning:" in res.stderr
    assert str(missing) in res.stderr


def test_idata_cli_validate_require_with_missing_explicit_bayesite_fails(
    bayesite_run: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-bayesite"
    out = tmp_path / "fit.nc"

    res = CliRunner().invoke(
        idata_cli,
        [
            str(bayesite_run),
            "-o",
            str(out),
            "--validate",
            "require",
            "--bayesite",
            str(missing),
        ],
    )

    assert res.exit_code == 2
    assert str(missing) in res.stderr
    assert "bayesite" in res.stderr


def test_idata_cli_validate_require_reports_diagnose_failure(
    bayesite_run: Path, tmp_path: Path
) -> None:
    exe = tmp_path / "bad-bayesite"
    log = tmp_path / "bad-args.txt"
    _write_fake_bayesite(exe, log, exit_code=17, stderr="diagnose boom")
    out = tmp_path / "fit.nc"

    res = CliRunner().invoke(
        idata_cli,
        [
            str(bayesite_run),
            "-o",
            str(out),
            "--validate",
            "require",
            "--bayesite",
            str(exe),
        ],
    )

    assert res.exit_code == 2
    assert "diagnose boom" in res.stderr
