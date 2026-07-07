"""Posterior draw artifact writer for the bayescycle run-directory v0 contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, SupportsFloat, cast

from bayesjax.diagnostics import ess, rhat
from bayesjax.model.bound import BoundModel
from bayeswire.ir import model_data_fingerprint as model_data_fingerprint
from jax import Array


class PosteriorArtifactError(RuntimeError):
    """Raised when sampler results cannot be serialized as a posterior artifact."""


class _ArrayLike(Protocol):
    """Small array surface used by the artifact writer."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, index: int) -> _ArrayLike: ...

    def __bool__(self) -> bool: ...

    def __float__(self) -> float: ...

    def __int__(self) -> int: ...

    def reshape(self, *shape: int) -> _ArrayLike: ...

    def tolist(self) -> object: ...


class _NutsDiagnosticTrace(Protocol):
    is_divergent: _ArrayLike
    acceptance_rate: _ArrayLike
    num_trajectory_expansions: _ArrayLike
    energy: _ArrayLike


class _SamplerDiagnostics(Protocol):
    sampling: _NutsDiagnosticTrace


class _SamplerAdaptation(Protocol):
    step_size: _ArrayLike


class _SamplerSettings(Protocol):
    max_tree_depth: int


class _SamplerResult(Protocol):
    samples: Mapping[str, _ArrayLike]
    diagnostics: _SamplerDiagnostics
    adaptation: _SamplerAdaptation
    settings: _SamplerSettings


type JsonValue = object

WORKFLOW_PHASES = [
    "parse_json",
    "decode_ir",
    "bind_data",
    "build_posterior_state",
    "evaluate_logp_grad",
    "run_nuts",
    "emit_artifact",
]


@dataclass(frozen=True)
class PosteriorArtifactSettings:
    """Sampler settings reported in posterior artifacts."""

    seed: int
    chains: int
    warmup: int
    draws: int
    max_tree_depth: int
    target_accept: float


def write_posterior_ndjson(
    path: Path,
    *,
    bound: BoundModel,
    result: _SamplerResult,
    settings: PosteriorArtifactSettings,
    fingerprint: str,
) -> None:
    """Write a v0 posterior draw stream for an in-process sampler result."""
    if not bound.param_shapes:
        raise PosteriorArtifactError(
            "bayesjax backend cannot write posterior.ndjson for a parameterless model; "
            "the v0 posterior artifact requires at least one free value"
        )
    _validate_result_shapes(bound, result, settings)
    lines = _posterior_documents(bound, result, settings, fingerprint)
    with path.open("x", encoding="utf-8") as f:
        for document in lines:
            f.write(json.dumps(document, separators=(",", ":"), allow_nan=False))
            f.write("\n")


def _posterior_documents(
    bound: BoundModel,
    result: _SamplerResult,
    settings: PosteriorArtifactSettings,
    fingerprint: str,
) -> list[JsonValue]:
    parameter_order = list(bound.param_shapes)
    chain_order = list(range(settings.chains))
    draw_count = settings.chains * settings.draws
    documents: list[JsonValue] = [
        _header(bound, settings, fingerprint, parameter_order, chain_order, draw_count)
    ]
    draw_index = 0
    for chain in chain_order:
        for draw in range(settings.draws):
            documents.append(
                _draw_line(
                    bound,
                    result,
                    settings,
                    parameter_order,
                    chain_order,
                    draw_count,
                    draw_index,
                    chain,
                    draw,
                )
            )
            draw_index += 1
    documents.append(
        {"trailer": _trailer(bound, result, settings, fingerprint, parameter_order, chain_order)}
    )
    return documents


def _header(
    bound: BoundModel,
    settings: PosteriorArtifactSettings,
    fingerprint: str,
    parameter_order: list[str],
    chain_order: list[int],
    draw_count: int,
) -> dict[str, JsonValue]:
    return {
        "draws_format": "v0-provisional",
        "artifact_kind": "posterior_draws",
        "artifact_scope": "observed_data_conditioned_parameter_draws",
        "workflow_phases": list(WORKFLOW_PHASES),
        "model_data_fingerprint": fingerprint,
        "params": [
            {
                "name": name,
                "shape": list(bound.param_shapes[name]),
                "coordinate_order": _coordinate_order(bound.param_shapes[name]),
            }
            for name in parameter_order
        ],
        "parameter_count": len(parameter_order),
        "parameter_order": list(parameter_order),
        "packing": list(parameter_order),
        "settings": {
            "num_warmup": settings.warmup,
            "num_draws": settings.draws,
            "max_treedepth": settings.max_tree_depth,
            "target_accept": settings.target_accept,
        },
        "seed": settings.seed,
        "chain_count": settings.chains,
        "chain_order": chain_order,
        "draw_count": draw_count,
        "chains": settings.chains,
        "sample_stats_mode": "per_draw_v2",
    }


def _draw_line(
    bound: BoundModel,
    result: _SamplerResult,
    settings: PosteriorArtifactSettings,
    parameter_order: list[str],
    chain_order: list[int],
    draw_count: int,
    draw_index: int,
    chain: int,
    draw: int,
) -> dict[str, JsonValue]:
    return {
        "draws_format": "v0-provisional",
        "artifact_kind": "posterior_draws",
        "artifact_scope": "observed_data_conditioned_parameter_draws",
        "draw_index": draw_index,
        "draw_index_base": "zero_based_retained_draw_order",
        "seed": settings.seed,
        "draw_count": draw_count,
        "chain_count": settings.chains,
        "chain_order": chain_order,
        "chain": chain,
        "chain_index_base": "zero_based_chain_id",
        "draw": draw,
        "parameter_count": len(parameter_order),
        "parameter_order": list(parameter_order),
        "values": {
            name: _parameter_value(result.samples[name], bound.param_shapes[name], chain, draw)
            for name in parameter_order
        },
        "sample_stats_mode": "per_draw_v2",
        "diverging": _bool_at(result.diagnostics.sampling.is_divergent, chain, draw),
        "tree_depth": _int_at(result.diagnostics.sampling.num_trajectory_expansions, chain, draw),
        "tree_accept": _probability_at(result.diagnostics.sampling.acceptance_rate, chain, draw),
        "energy": _finite_float_at(result.diagnostics.sampling.energy, chain, draw),
    }


def _trailer(
    bound: BoundModel,
    result: _SamplerResult,
    settings: PosteriorArtifactSettings,
    fingerprint: str,
    parameter_order: list[str],
    chain_order: list[int],
) -> dict[str, JsonValue]:
    sample_map = cast(dict[str, Array], dict(result.samples))
    return {
        "draws_format": "v0-provisional",
        "artifact_kind": "posterior_draws",
        "artifact_scope": "observed_data_conditioned_parameter_draws",
        "workflow_phases": list(WORKFLOW_PHASES),
        "model_data_fingerprint": fingerprint,
        "seed": settings.seed,
        "draws_per_chain": settings.draws,
        "chain_count": settings.chains,
        "chain_order": chain_order,
        "draw_count": settings.chains * settings.draws,
        "parameter_count": len(parameter_order),
        "parameter_order": list(parameter_order),
        "params": len(parameter_order),
        "chains": [_chain_stats(result, settings, chain) for chain in chain_order],
        "rhat": _diagnostic_map(rhat(sample_map), parameter_order),
        "ess": _diagnostic_map(ess(sample_map), parameter_order),
    }


def _chain_stats(
    result: _SamplerResult,
    settings: PosteriorArtifactSettings,
    chain: int,
) -> dict[str, JsonValue]:
    depths = [
        _int_at(result.diagnostics.sampling.num_trajectory_expansions, chain, draw)
        for draw in range(settings.draws)
    ]
    histogram = [0] * (settings.max_tree_depth + 1)
    for depth in depths:
        if depth < 0 or depth > settings.max_tree_depth:
            raise PosteriorArtifactError(
                f"bayesjax reported tree depth {depth} outside configured max_tree_depth "
                f"{settings.max_tree_depth}"
            )
        histogram[depth] += 1
    accepts = [
        _probability_at(result.diagnostics.sampling.acceptance_rate, chain, draw)
        for draw in range(settings.draws)
    ]
    return {
        "chain": chain,
        "chain_index_base": "zero_based_chain_id",
        "draw_count": settings.draws,
        "divergences": sum(
            1
            for draw in range(settings.draws)
            if _bool_at(result.diagnostics.sampling.is_divergent, chain, draw)
        ),
        "treedepth_histogram": histogram,
        "step_size": _finite_positive_float(_scalar_from_index(result.adaptation.step_size, chain)),
        "mean_accept": sum(accepts) / len(accepts),
    }


def _validate_result_shapes(
    bound: BoundModel,
    result: _SamplerResult,
    settings: PosteriorArtifactSettings,
) -> None:
    if result.settings.max_tree_depth != settings.max_tree_depth:
        raise PosteriorArtifactError(
            "bayesjax result max_tree_depth does not match requested artifact settings"
        )
    for name, shape in bound.param_shapes.items():
        expected = (settings.chains, settings.draws, *shape)
        got = tuple(result.samples[name].shape)
        if got != expected:
            raise PosteriorArtifactError(
                f"sample array for {name!r} has shape {got}; expected {expected}"
            )
    for field_name, array in (
        ("is_divergent", result.diagnostics.sampling.is_divergent),
        ("acceptance_rate", result.diagnostics.sampling.acceptance_rate),
        ("num_trajectory_expansions", result.diagnostics.sampling.num_trajectory_expansions),
        ("energy", result.diagnostics.sampling.energy),
    ):
        got = tuple(array.shape)
        expected = (settings.chains, settings.draws)
        if got != expected:
            raise PosteriorArtifactError(
                f"diagnostic {field_name} has shape {got}; expected {expected}"
            )
    if tuple(result.adaptation.step_size.shape) != (settings.chains,):
        raise PosteriorArtifactError(
            "adapted step_size has shape "
            f"{tuple(result.adaptation.step_size.shape)}; expected {(settings.chains,)}"
        )


def _coordinate_order(shape: tuple[int, ...]) -> list[list[int]]:
    if 0 in shape:
        return []
    size = math.prod(shape) if shape else 1
    coordinates: list[list[int]] = []
    for flat in range(size):
        remainder = flat
        coordinate = [0] * len(shape)
        for axis in range(len(shape) - 1, -1, -1):
            dim = shape[axis]
            coordinate[axis] = remainder % dim
            remainder //= dim
        coordinates.append(coordinate)
    return coordinates


def _parameter_value(array: _ArrayLike, shape: tuple[int, ...], chain: int, draw: int) -> JsonValue:
    value = _scalar_from_index(array, chain, draw)
    if shape == ():
        return _finite_float(value)
    flattened = value.reshape(math.prod(shape)).tolist()
    if not isinstance(flattened, list):
        raise PosteriorArtifactError("non-scalar parameter value did not flatten to a list")
    return [_finite_float(cast(SupportsFloat, item)) for item in flattened]


def _scalar_from_index(array: _ArrayLike, *index: int) -> _ArrayLike:
    value = array
    for item in index:
        value = value[item]
    return value


def _bool_at(array: _ArrayLike, chain: int, draw: int) -> bool:
    return bool(_scalar_from_index(array, chain, draw))


def _int_at(array: _ArrayLike, chain: int, draw: int) -> int:
    return int(_scalar_from_index(array, chain, draw))


def _finite_float_at(array: _ArrayLike, chain: int, draw: int) -> float:
    return _finite_float(_scalar_from_index(array, chain, draw))


def _probability_at(array: _ArrayLike, chain: int, draw: int) -> float:
    return _probability_float(_scalar_from_index(array, chain, draw))


def _finite_float(value: SupportsFloat) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise PosteriorArtifactError("posterior artifact values must be finite")
    return number


def _finite_positive_float(value: SupportsFloat) -> float:
    number = _finite_float(value)
    if number <= 0.0:
        raise PosteriorArtifactError("adapted step_size must be positive")
    return number


def _probability_float(value: SupportsFloat) -> float:
    number = _finite_float(value)
    tolerance = 1e-6
    if -tolerance <= number < 0.0:
        return 0.0
    if 1.0 < number <= 1.0 + tolerance:
        return 1.0
    if number < 0.0 or number > 1.0:
        raise PosteriorArtifactError("acceptance probability must be in [0, 1]")
    return number


def _diagnostic_map(
    values: Mapping[str, float], parameter_order: list[str]
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for name in parameter_order:
        value = values.get(name)
        result[name] = None if value is None or not math.isfinite(value) else float(value)
    return result
