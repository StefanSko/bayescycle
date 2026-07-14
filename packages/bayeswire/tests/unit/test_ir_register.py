"""Unit tests for registering user distributions as IR extension tags."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass, field
from typing import cast

import pytest

from bayeswire.distributions.core import (
    Distribution,
    DistributionParameter,
    DistributionValue,
    LogProbability,
)
from bayeswire.ir import (
    UnserializableDistribution,
    UnserializableValue,
    meta_from_dict,
    meta_to_dict,
    register_distribution,
)
from bayeswire.model.decorator import (
    ModelMeta,
    ResolvedFreeValue,
    ResolvedParam,
    ResolvedStochasticSite,
)
from bayeswire.model.expr import DataRef, ParamRef


@dataclass(frozen=True)
class _Gumbel:
    """Custom location-scale distribution used as a registry extension."""

    loc: DistributionParameter
    scale: DistributionParameter

    def log_prob(self, x: DistributionValue) -> LogProbability:
        raise NotImplementedError


@dataclass(frozen=True)
class _RenamedLaplace:
    """Distribution registered under an explicit wire tag."""

    loc: DistributionParameter

    def log_prob(self, x: DistributionValue) -> LogProbability:
        raise NotImplementedError


@dataclass(frozen=True)
class _MapDistribution:
    """Extension used to verify nested map values remain wire values."""

    values: dict[str, object]


@dataclass(frozen=True)
class _TupleDistribution:
    """Extension used to verify registered tuple field kinds at runtime."""

    values: tuple[object, ...]


@dataclass(frozen=True, init=False)
class _NoInitDistribution:
    """Invalid extension whose class constructor accepts no encoded fields."""

    loc: object = 0.0


@dataclass(frozen=True)
class _ExplicitInitDistribution:
    """Invalid extension with a constructor incompatible with its encoded field."""

    loc: object

    def __init__(self) -> None:
        object.__setattr__(self, "loc", 0.0)


@dataclass(frozen=True)
class _CachedDistribution:
    """Invalid extension with state excluded from its constructor."""

    loc: object
    cache: object = field(init=False, default=1.0)


@dataclass(frozen=True)
class _IgnoredFieldDistribution:
    """Invalid extension whose constructor field is excluded from equality."""

    marker: int = field(compare=False)


@dataclass(frozen=True, eq=False)
class _IdentityDistribution:
    """Invalid extension whose decoded copies cannot compare structurally."""

    loc: object


@dataclass
class _MutableDistribution:
    """Invalid mutable extension used to protect metadata immutability."""

    loc: object


class _PlainDistribution:
    """Distribution without dataclass fields."""

    def log_prob(self, x: DistributionValue) -> LogProbability:
        raise NotImplementedError


def _meta_with(distribution: Distribution) -> ModelMeta:
    return ModelMeta(
        params={"alpha": ResolvedParam(distribution, constraint=None, size=None)},
        data={},
        observed_nodes=(),
        expressions={},
        free_values={"alpha": ResolvedFreeValue(constraint=None, size=None)},
        stochastic_sites=(ResolvedStochasticSite("alpha", distribution, ParamRef("alpha")),),
    )


def test_registered_distribution_round_trips_with_symbolic_fields() -> None:
    register_distribution(_Gumbel)
    meta = _meta_with(_Gumbel(DataRef("mu0"), 2.0))

    document = meta_to_dict(meta)

    model = document["model"]
    assert isinstance(model, dict)
    params = model["params"]
    assert isinstance(params, list)
    entry = params[0]
    assert isinstance(entry, dict)
    value = entry["value"]
    assert isinstance(value, dict)
    assert value["distribution"] == {
        "node": "_Gumbel",
        "loc": {"node": "DataRef", "name": "mu0"},
        "scale": 2.0,
    }
    assert meta_from_dict(document) == meta


def test_register_distribution_is_idempotent() -> None:
    register_distribution(_Gumbel)
    register_distribution(_Gumbel)

    assert meta_from_dict(meta_to_dict(_meta_with(_Gumbel(0.0, 1.0)))) is not None


def test_registered_distribution_rejects_runtime_field_kind_mismatches() -> None:
    register_distribution(_TupleDistribution)
    malformed = _TupleDistribution(cast(tuple[object, ...], {"x": 1}))

    with pytest.raises(UnserializableValue, match=r"_TupleDistribution.*values.*tuple"):
        meta_to_dict(_meta_with(malformed))


@pytest.mark.parametrize("nested", [{"x": 1}, (1, 2)])
def test_registered_distribution_rejects_nested_bare_containers(nested: object) -> None:
    register_distribution(_MapDistribution)

    with pytest.raises(UnserializableValue, match=r"bare (dict|tuple)"):
        meta_to_dict(_meta_with(_MapDistribution({"nested": nested})))


@pytest.mark.parametrize("cls", [_NoInitDistribution, _ExplicitInitDistribution])
def test_register_distribution_rejects_incompatible_class_constructors(cls: type) -> None:
    with pytest.raises(UnserializableDistribution, match=rf"{cls.__name__}.*constructor"):
        register_distribution(cls)


def test_register_distribution_rejects_nonconstructor_fields() -> None:
    with pytest.raises(UnserializableDistribution, match=r"_CachedDistribution.*init=True"):
        register_distribution(_CachedDistribution)


def test_register_distribution_rejects_noncomparing_constructor_fields() -> None:
    with pytest.raises(
        UnserializableDistribution, match=r"_IgnoredFieldDistribution.*compare=True"
    ):
        register_distribution(_IgnoredFieldDistribution)


def test_register_distribution_requires_structural_dataclass_equality() -> None:
    with pytest.raises(UnserializableDistribution, match=r"_IdentityDistribution.*eq=True"):
        register_distribution(_IdentityDistribution)


def test_register_distribution_rejects_mutable_dataclasses() -> None:
    with pytest.raises(UnserializableDistribution, match=r"_MutableDistribution.*frozen=True"):
        register_distribution(_MutableDistribution)


def test_register_distribution_rejects_non_dataclass_with_repair_instruction() -> None:
    with pytest.raises(
        UnserializableDistribution,
        match=r"_PlainDistribution.*dataclass",
    ):
        register_distribution(_PlainDistribution)


def test_register_distribution_cannot_reclassify_constraint_nodes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from bayeswire.constraints import Positive
from bayeswire.ir import UnserializableDistribution, register_distribution

try:
    register_distribution(Positive)
except UnserializableDistribution:
    pass
else:
    raise AssertionError("Positive was reclassified as a distribution")
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_register_distribution_supports_explicit_tag_override() -> None:
    register_distribution(_RenamedLaplace, tag="Laplace")
    meta = _meta_with(_RenamedLaplace(0.5))

    document = meta_to_dict(meta)

    model = document["model"]
    assert isinstance(model, dict)
    params = model["params"]
    assert isinstance(params, list)
    entry = params[0]
    assert isinstance(entry, dict)
    value = entry["value"]
    assert isinstance(value, dict)
    assert value["distribution"] == {"node": "Laplace", "loc": 0.5}
    assert meta_from_dict(document) == meta


def test_register_distribution_rejects_tag_collisions() -> None:
    @dataclass(frozen=True)
    class Normal:
        rate: float

        def log_prob(self, x: DistributionValue) -> LogProbability:
            raise NotImplementedError

    with pytest.raises(ValueError, match="'Normal' is already registered"):
        register_distribution(Normal)


def test_registered_distribution_still_rejects_non_finite_fields() -> None:
    register_distribution(_Gumbel)

    from bayeswire.ir import NonFiniteConstant

    with pytest.raises(NonFiniteConstant):
        meta_to_dict(_meta_with(_Gumbel(math.inf, 1.0)))
