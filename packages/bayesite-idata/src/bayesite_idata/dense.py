"""Checked dense fit representation between run artifacts and ArviZ.

This module owns the exporter invariants. Parsed protocol/data/model facts are
materialized here first; ArviZ conversion is a final adapter step in
``bayesite_idata.arviz``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from bayesite_idata.data_json import read_data_arrays
from bayesite_idata.dims_json import CoordValue, DimsSidecar
from bayesite_idata.errors import AssemblyError
from bayesite_idata.model_ir import observed_data_names
from bayesite_idata.protocol_v0 import (
    PER_DRAW_SAMPLE_STATS_V1,
    PER_DRAW_SAMPLE_STATS_V2,
    AttrValue,
    DrawRecord,
    NumberValue,
    PosteriorPredictiveStream,
    PosteriorStream,
    PriorPredictiveRecord,
    PriorPredictiveStream,
    SiteSpec,
)

type NumericArray = npt.NDArray[np.integer | np.floating | np.bool_]
type CoordValues = tuple[CoordValue, ...]


@dataclass(frozen=True)
class DenseVar:
    """One named dense variable with explicit dims."""

    name: str
    values: NumericArray
    dims: tuple[str, ...]


@dataclass(frozen=True)
class DenseGroup:
    """One ArviZ group before the ArviZ adapter sees it."""

    name: str
    variables: tuple[DenseVar, ...]
    coords: Mapping[str, CoordValues]
    sample_dims: tuple[str, ...] = ()

    @property
    def var_names(self) -> tuple[str, ...]:
        return tuple(var.name for var in self.variables)

    def require_var(self, name: str) -> DenseVar:
        for var in self.variables:
            if var.name == name:
                return var
        msg = f"dense group {self.name!r} has no variable {name!r}"
        raise AssemblyError(msg)


@dataclass(frozen=True)
class CheckedDenseFit:
    """Complete checked dense fit ready for ArviZ conversion."""

    groups: tuple[DenseGroup, ...]
    posterior_attrs: Mapping[str, AttrValue]

    def require_group(self, name: str) -> DenseGroup:
        for group in self.groups:
            if group.name == name:
                return group
        msg = f"dense fit has no group {name!r}"
        raise AssemblyError(msg)

    def group(self, name: str) -> DenseGroup | None:
        for group in self.groups:
            if group.name == name:
                return group
        return None

    def without_group(self, name: str) -> CheckedDenseFit:
        return replace(self, groups=tuple(group for group in self.groups if group.name != name))


def build_checked_dense_fit(
    *,
    model_ir: Path,
    data_json: Path,
    posterior: PosteriorStream,
    prior_predictive: PriorPredictiveStream | None = None,
    posterior_predictive: PosteriorPredictiveStream | None = None,
    dims_sidecar: DimsSidecar | None = None,
) -> CheckedDenseFit:
    """Build and validate the dense intermediate fit."""
    observed_names = _observed_names(model_ir, posterior_predictive)
    observed_data, constant_data = _data_groups(data_json, observed_names)

    groups: list[DenseGroup] = [
        _posterior_group(posterior, dims_sidecar),
        _data_group("observed_data", observed_data, dims_sidecar),
    ]
    if constant_data:
        groups.append(_data_group("constant_data", constant_data, dims_sidecar))
    sample_stats = _sample_stats_group(posterior)
    if sample_stats is not None:
        groups.append(sample_stats)
    if posterior_predictive is not None:
        groups.append(_posterior_predictive_group(posterior, posterior_predictive, dims_sidecar))
    if prior_predictive is not None:
        prior, prior_pred = _prior_predictive_groups(prior_predictive, dims_sidecar)
        if prior is not None:
            groups.append(prior)
        if prior_pred is not None:
            groups.append(prior_pred)

    fit = CheckedDenseFit(groups=tuple(groups), posterior_attrs=posterior.attrs)
    validate_dense_fit(fit)
    return fit


def validate_dense_fit(fit: CheckedDenseFit) -> None:
    """Validate dense fit invariants before ArviZ conversion."""
    fit.require_group("posterior")
    for group in fit.groups:
        coord_var_conflicts = sorted(set(group.coords) & set(group.var_names))
        if coord_var_conflicts:
            msg = (
                f"{group.name} coordinate name(s) conflict with variable name(s): "
                f"{coord_var_conflicts}"
            )
            raise AssemblyError(msg)
        for var in group.variables:
            if len(var.dims) != var.values.ndim:
                msg = f"{group.name}.{var.name} dims do not match array rank"
                raise AssemblyError(msg)
            if group.sample_dims and var.dims[: len(group.sample_dims)] != group.sample_dims:
                msg = f"{group.name}.{var.name} does not start with sample dims {group.sample_dims}"
                raise AssemblyError(msg)
            for axis, dim in enumerate(var.dims):
                if dim in group.coords and len(group.coords[dim]) != var.values.shape[axis]:
                    msg = f"{group.name}.{var.name} coord {dim!r} length does not match axis"
                    raise AssemblyError(msg)

    posterior_predictive = fit.group("posterior_predictive")
    if posterior_predictive is not None:
        observed_data = fit.group("observed_data")
        if observed_data is None:
            msg = "posterior_predictive requires matching observed_data"
            raise AssemblyError(msg)
        missing = sorted(set(posterior_predictive.var_names) - set(observed_data.var_names))
        if missing:
            msg = f"posterior_predictive variable(s) missing observed_data: {missing}"
            raise AssemblyError(msg)
        for pred_var in posterior_predictive.variables:
            observed_var = observed_data.require_var(pred_var.name)
            event_shape = pred_var.values.shape[len(posterior_predictive.sample_dims) :]
            if tuple(event_shape) != tuple(observed_var.values.shape):
                msg = (
                    f"posterior_predictive variable {pred_var.name!r} shape {tuple(event_shape)} "
                    f"does not match observed_data shape {tuple(observed_var.values.shape)}"
                )
                raise AssemblyError(msg)


def _observed_names(
    model_ir: Path, posterior_predictive: PosteriorPredictiveStream | None
) -> frozenset[str] | None:
    if posterior_predictive is not None:
        return frozenset(site.name for site in posterior_predictive.sites)
    return observed_data_names(model_ir)


def _posterior_group(posterior: PosteriorStream, dims_sidecar: DimsSidecar | None) -> DenseGroup:
    index = _record_index(posterior)
    variables: list[DenseVar] = []
    coords = _sample_coords(posterior.chains, posterior.draws_per_chain)
    for param in posterior.params:
        shape = (len(posterior.chains), posterior.draws_per_chain, *param.shape)
        arr = np.empty(shape, dtype=np.float64)
        for chain_i, chain in enumerate(posterior.chains):
            for draw in range(posterior.draws_per_chain):
                arr[(chain_i, draw, *([slice(None)] * len(param.shape)))] = np.asarray(
                    index[(chain, draw)].values[param.name], dtype=np.float64
                ).reshape(param.shape)
        extra_dims = _event_dims(param.name, param.shape, dims_sidecar)
        _merge_coords(coords, _coords_for_dims(param.name, extra_dims, param.shape, dims_sidecar))
        variables.append(DenseVar(param.name, arr, ("chain", "draw", *extra_dims)))
    return DenseGroup(
        name="posterior",
        variables=tuple(variables),
        coords=MappingProxyType(coords),
        sample_dims=("chain", "draw"),
    )


def _sample_stats_group(posterior: PosteriorStream) -> DenseGroup | None:
    if posterior.sample_stats_mode not in {PER_DRAW_SAMPLE_STATS_V1, PER_DRAW_SAMPLE_STATS_V2}:
        return None
    index = _record_index(posterior)
    shape = (len(posterior.chains), posterior.draws_per_chain)
    diverging = np.empty(shape, dtype=np.bool_)
    tree_depth = np.empty(shape, dtype=np.int64)
    acceptance_rate = np.empty(shape, dtype=np.float64)
    energy = (
        np.empty(shape, dtype=np.float64)
        if posterior.sample_stats_mode == PER_DRAW_SAMPLE_STATS_V2
        else None
    )
    for chain_i, chain in enumerate(posterior.chains):
        for draw in range(posterior.draws_per_chain):
            record = index[(chain, draw)]
            if record.diverging is None or record.tree_depth is None or record.tree_accept is None:
                msg = "per-draw sample stats announced but a parsed record is missing stats"
                raise AssemblyError(msg)
            diverging[chain_i, draw] = record.diverging
            tree_depth[chain_i, draw] = record.tree_depth
            acceptance_rate[chain_i, draw] = record.tree_accept
            if energy is not None:
                if record.energy is None:
                    msg = "per-draw v2 sample stats announced but a parsed record is missing energy"
                    raise AssemblyError(msg)
                energy[chain_i, draw] = record.energy
    variables = [
        DenseVar("diverging", diverging, ("chain", "draw")),
        DenseVar("tree_depth", tree_depth, ("chain", "draw")),
        DenseVar("acceptance_rate", acceptance_rate, ("chain", "draw")),
    ]
    if energy is not None:
        variables.append(DenseVar("energy", energy, ("chain", "draw")))
    return DenseGroup(
        name="sample_stats",
        variables=tuple(variables),
        coords=MappingProxyType(_sample_coords(posterior.chains, posterior.draws_per_chain)),
        sample_dims=("chain", "draw"),
    )


def _data_groups(
    path: Path, observed_names: frozenset[str] | None
) -> tuple[Mapping[str, NumericArray], Mapping[str, NumericArray]]:
    arrays = read_data_arrays(path)
    if observed_names is not None:
        missing = sorted(observed_names - set(arrays))
        if missing:
            msg = f"{path}: missing observed data value(s) required by predictive sites: {missing}"
            raise AssemblyError(msg)
    observed: dict[str, NumericArray] = {}
    constant: dict[str, NumericArray] = {}
    for name, arr in arrays.items():
        if observed_names is None or name in observed_names:
            observed[name] = arr
        else:
            constant[name] = arr
    return MappingProxyType(observed), MappingProxyType(constant)


def _data_group(
    name: str, arrays: Mapping[str, NumericArray], dims_sidecar: DimsSidecar | None
) -> DenseGroup:
    variables: list[DenseVar] = []
    coords: dict[str, CoordValues] = {}
    for var_name, arr in arrays.items():
        shape = tuple(int(dim) for dim in arr.shape)
        dims = _event_dims(var_name, shape, dims_sidecar)
        _merge_coords(coords, _coords_for_dims(var_name, dims, shape, dims_sidecar))
        variables.append(DenseVar(var_name, arr, dims))
    return DenseGroup(name=name, variables=tuple(variables), coords=MappingProxyType(coords))


def _prior_predictive_groups(
    stream: PriorPredictiveStream, dims_sidecar: DimsSidecar | None
) -> tuple[DenseGroup | None, DenseGroup | None]:
    prior_vars: list[DenseVar] = []
    pred_vars: list[DenseVar] = []
    records = sorted(stream.records, key=lambda record: record.draw)
    prior_coords = _sample_coords((0,), stream.draws)
    pred_coords = _sample_coords((0,), stream.draws)
    for site in stream.sites:
        arr = _prior_predictive_array(site, records)
        extra_dims = _event_dims(site.name, site.shape, dims_sidecar)
        event_coords = _coords_for_dims(site.name, extra_dims, site.shape, dims_sidecar)
        var = DenseVar(site.name, arr, ("chain", "draw", *extra_dims))
        if site.role == "parameter":
            _merge_coords(prior_coords, event_coords)
            prior_vars.append(var)
        else:
            _merge_coords(pred_coords, event_coords)
            pred_vars.append(var)
    prior = _sample_group("prior", prior_vars, prior_coords) if prior_vars else None
    prior_pred = _sample_group("prior_predictive", pred_vars, pred_coords) if pred_vars else None
    return prior, prior_pred


def _prior_predictive_array(
    site: SiteSpec, records: Sequence[PriorPredictiveRecord]
) -> NumericArray:
    dtype = _site_dtype(site, (record.values[site.name] for record in records))
    arr = np.empty((1, len(records), *site.shape), dtype=dtype)
    for i, record in enumerate(records):
        arr[(0, i, *([slice(None)] * len(site.shape)))] = np.asarray(
            record.values[site.name], dtype=dtype
        ).reshape(site.shape)
    return arr


def _posterior_predictive_group(
    posterior: PosteriorStream,
    stream: PosteriorPredictiveStream,
    dims_sidecar: DimsSidecar | None,
) -> DenseGroup:
    posterior_seed = posterior.attrs.get("seed")
    if (
        stream.source_fit_seed is not None
        and posterior_seed is not None
        and stream.source_fit_seed != posterior_seed
    ):
        msg = (
            "posterior_predictive source_fit_seed does not match posterior seed: "
            f"{stream.source_fit_seed} != {posterior_seed}"
        )
        raise AssemblyError(msg)
    chain_index = {chain: i for i, chain in enumerate(posterior.chains)}
    expected = {
        (chain, draw) for chain in posterior.chains for draw in range(posterior.draws_per_chain)
    }
    seen = {(record.source_chain, record.source_draw) for record in stream.records}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        msg = (
            "posterior_predictive source cells do not match posterior; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
        raise AssemblyError(msg)
    records = {(record.source_chain, record.source_draw): record for record in stream.records}
    variables: list[DenseVar] = []
    coords = _sample_coords(posterior.chains, posterior.draws_per_chain)
    for site in stream.sites:
        dtype = _site_dtype(site, (record.values[site.name] for record in stream.records))
        arr = np.empty((len(posterior.chains), posterior.draws_per_chain, *site.shape), dtype=dtype)
        for chain in posterior.chains:
            for draw in range(posterior.draws_per_chain):
                record = records[(chain, draw)]
                arr[(chain_index[chain], draw, *([slice(None)] * len(site.shape)))] = np.asarray(
                    record.values[site.name], dtype=dtype
                ).reshape(site.shape)
        extra_dims = _event_dims(site.name, site.shape, dims_sidecar)
        _merge_coords(coords, _coords_for_dims(site.name, extra_dims, site.shape, dims_sidecar))
        variables.append(DenseVar(site.name, arr, ("chain", "draw", *extra_dims)))
    return _sample_group("posterior_predictive", variables, coords)


def _sample_group(
    name: str, variables: Sequence[DenseVar], coords: Mapping[str, CoordValues]
) -> DenseGroup:
    return DenseGroup(
        name=name,
        variables=tuple(variables),
        coords=MappingProxyType(dict(coords)),
        sample_dims=("chain", "draw"),
    )


def _record_index(posterior: PosteriorStream) -> Mapping[tuple[int, int], DrawRecord]:
    return {(record.chain, record.draw): record for record in posterior.records}


def _sample_coords(chains: Sequence[int], draws_per_chain: int) -> dict[str, CoordValues]:
    return {"chain": tuple(chains), "draw": tuple(range(draws_per_chain))}


def _event_dims(
    variable: str, shape: tuple[int, ...], dims_sidecar: DimsSidecar | None
) -> tuple[str, ...]:
    if dims_sidecar is None:
        return _extra_dims(variable, shape)
    declared = dims_sidecar.dims_for(variable)
    if declared is None:
        return _extra_dims(variable, shape)
    if len(declared) != len(shape):
        msg = (
            f"dims.json declares rank {len(declared)} for variable {variable!r}, "
            f"but artifact shape rank is {len(shape)}"
        )
        raise AssemblyError(msg)
    return declared


def _extra_dims(name: str, shape: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"{name}_dim_{axis}" for axis, _size in enumerate(shape))


def _coords_for_dims(
    variable: str,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    dims_sidecar: DimsSidecar | None,
) -> dict[str, CoordValues]:
    coords: dict[str, CoordValues] = {}
    for dim, size in zip(dims, shape, strict=True):
        declared = dims_sidecar.coords_for(dim) if dims_sidecar is not None else None
        if declared is not None:
            if len(declared) != size:
                msg = (
                    f"dims.json coord {dim!r} length {len(declared)} does not match "
                    f"variable {variable!r} axis size {size}"
                )
                raise AssemblyError(msg)
            coords[dim] = declared
        else:
            coords[dim] = tuple(range(size))
    return coords


def _merge_coords(target: dict[str, CoordValues], extra: Mapping[str, CoordValues]) -> None:
    for dim, values in extra.items():
        existing = target.get(dim)
        if existing is not None and existing != values:
            msg = f"coordinate {dim!r} has conflicting values across variables"
            raise AssemblyError(msg)
        target[dim] = values


def _site_dtype(
    site: SiteSpec, value_rows: Iterable[tuple[NumberValue, ...]]
) -> type[np.int64] | type[np.float64]:
    if not site.integer:
        return np.float64
    for values in value_rows:
        if any(not _is_integer_value(value) for value in values):
            msg = f"generated site {site.name!r} is marked integer but has non-integer values"
            raise AssemblyError(msg)
    return np.int64


def _is_integer_value(value: NumberValue) -> bool:
    if isinstance(value, int):
        return True
    return value.is_integer()
