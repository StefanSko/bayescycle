"""Deterministic reference metadata pinning the bayeswire IR v1 wire format.

Most cases come directly from eDSL declarations; documented adversarial cases
exercise the resolved-ModelMeta boundary. Shared by the produce-conformance
tests and ``scripts/regenerate_corpus.py``. Any change to a corpus document is a
wire-format change and requires a
deliberate corpus diff, a spec changelog entry, and a version decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from bayeswire import Data, Dim, Observed, Param, PartiallyObserved, Submodel, model, with_prior
from bayeswire.constraints import Interval, Ordered, Positive, UnitInterval
from bayeswire.distributions import (
    Bernoulli,
    Beta,
    Exponential,
    HalfNormal,
    MultivariateNormal,
    Normal,
    OrderedLogistic,
    Poisson,
    Truncated,
)
from bayeswire.math import exp
from bayeswire.model.decorator import ModelMeta, ResolvedStochasticSite


@dataclass(frozen=True)
class ReferenceModelCase:
    """One golden model with deterministic bind values."""

    name: str
    model_cls: type
    meta: ModelMeta
    bind_values: dict[str, object]


def _meta(cls: type) -> ModelMeta:
    return cast(ModelMeta, getattr(cls, "_model_meta"))  # noqa: B009


def _linear_regression() -> ReferenceModelCase:
    @model
    class LinearRegression:
        alpha = Param(Normal(0.0, 1.0))
        beta = Param(Normal(0.0, 1.0))
        sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())
        x = Data.vector()
        mu = alpha + beta * x
        y = Observed(Normal(mu, sigma))

    return ReferenceModelCase(
        name="linear_regression",
        model_cls=LinearRegression,
        meta=_meta(LinearRegression),
        bind_values={
            "x": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "y": [-1.6, -0.3, 0.4, 1.1, 2.2],
        },
    )


def _alternative_prior_regression() -> ReferenceModelCase:
    @model
    class RegressionTarget:
        alpha = Param(Normal(0.0, 1.0))
        beta = Param(Normal(0.0, 1.0))
        sigma = Param(HalfNormal(1.0), constraint=Positive())
        x = Data.vector()
        mu = alpha + beta * x
        y = Observed(Normal(mu, sigma))

    @model
    class SimulationPrior:
        alpha = Param(Normal(-0.1, 0.2))
        beta = Param(Normal(1.25, 0.1))
        sigma = Param(HalfNormal(2.0), constraint=Positive())

    @model
    class HandwrittenAlternativeRegression:
        alpha = Param(Normal(-0.1, 0.2))
        beta = Param(Normal(1.25, 0.1))
        sigma = Param(HalfNormal(2.0), constraint=Positive())
        x = Data.vector()
        mu = alpha + beta * x
        y = Observed(Normal(mu, sigma))

    AlternativeRegression = with_prior(RegressionTarget, prior=SimulationPrior)
    assert _meta(AlternativeRegression) == _meta(HandwrittenAlternativeRegression)

    return ReferenceModelCase(
        name="alternative_prior_regression",
        model_cls=AlternativeRegression,
        meta=_meta(AlternativeRegression),
        bind_values={
            "x": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "y": [-1.6, -0.3, 0.4, 1.1, 2.2],
        },
    )


def _eight_schools_non_centered() -> ReferenceModelCase:
    # Dim labels are sidecar-only semantic metadata: they change the
    # dims.json artifact, never the IR document or its canonical bytes.
    school = Dim("school", coords=("A", "B", "C", "D", "E", "F", "G", "H"))

    @model
    class EightSchoolsNonCentered:
        n_schools = Data.scalar()
        sigma = Data.vector(n_schools, dims=(school,))

        mu = Param(Normal(0.0, 5.0))
        tau = Param(HalfNormal(5.0), constraint=Positive())
        z = Param(Normal(0.0, 1.0), size=n_schools, dims=(school,))
        theta = mu + tau * z
        y = Observed(Normal(theta, sigma), dims=(school,))

    return ReferenceModelCase(
        name="eight_schools_non_centered",
        model_cls=EightSchoolsNonCentered,
        meta=_meta(EightSchoolsNonCentered),
        bind_values={
            "n_schools": 8,
            "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
            "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        },
    )


def _mvn_non_centered() -> ReferenceModelCase:
    @model
    class MvnNonCentered:
        n = Data.scalar()
        mean = Data.vector(n)
        latent_chol = Data.matrix(n, n)
        observation_chol = Data.matrix(n, n)

        z = Param(Normal(0.0, 1.0), size=n)
        theta = mean + latent_chol @ z
        y = Observed(MultivariateNormal(theta, observation_chol))

    return ReferenceModelCase(
        name="mvn_non_centered",
        model_cls=MvnNonCentered,
        meta=_meta(MvnNonCentered),
        bind_values={
            "n": 3,
            "mean": [0.25, -0.5, 1.0],
            "latent_chol": [
                [1.0, 0.0, 0.0],
                [0.6, 0.8, 0.0],
                [0.25, 0.35, 0.9],
            ],
            "observation_chol": [
                [0.5, 0.0, 0.0],
                [0.1, 0.6, 0.0],
                [-0.05, 0.2, 0.7],
            ],
            "y": [0.4, -0.1, 0.65],
        },
    )


def _varying_intercepts_poisson() -> ReferenceModelCase:
    @model
    class VaryingInterceptsPoisson:
        n_groups = Data.scalar()
        group_idx = Data.vector()
        x = Data.vector()

        alpha_pop = Param(Normal(0.0, 0.5))
        sigma_alpha = Param(HalfNormal(0.4), constraint=Positive())
        z_alpha = Param(Normal(0.0, 1.0), size=n_groups)

        alpha = alpha_pop + sigma_alpha * z_alpha
        eta = alpha[group_idx] + 0.25 * x
        y = Observed(Poisson(exp(eta)))

    return ReferenceModelCase(
        name="varying_intercepts_poisson",
        model_cls=VaryingInterceptsPoisson,
        meta=_meta(VaryingInterceptsPoisson),
        bind_values={
            "n_groups": 3,
            "group_idx": [0, 0, 1, 1, 2, 2],
            "x": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            "y": [0, 1, 2, 1, 3, 2],
        },
    )


def _composed_measurements() -> ReferenceModelCase:
    @model
    class Measurement:
        offset = Data.scalar()
        location = Param(Normal(offset, 1.0))
        centered = location - offset
        values = Observed(Normal(centered, 1.0))

    @model
    class ComposedMeasurements:
        first = Submodel(Measurement)
        second = Submodel(Measurement)
        contrast = first.centered - second.centered
        comparison = Observed(Normal(contrast, 1.0))

    return ReferenceModelCase(
        name="composed_measurements",
        model_cls=ComposedMeasurements,
        meta=_meta(ComposedMeasurements),
        bind_values={
            "first.offset": 0.25,
            "first.values": [0.1, 0.4, -0.2],
            "second.offset": -0.5,
            "second.values": [-0.3, 0.2, 0.6],
            "comparison": 0.75,
        },
    )


def _ordinal_regression() -> ReferenceModelCase:
    @model
    class OrdinalRegression:
        n_cutpoints = Data.scalar()
        x = Data.vector()

        beta = Param(Normal(0.0, 1.0))
        cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())

        eta = beta * x
        y = Observed(OrderedLogistic(eta, cutpoints))

    return ReferenceModelCase(
        name="ordinal_regression",
        model_cls=OrdinalRegression,
        meta=_meta(OrdinalRegression),
        bind_values={
            "n_cutpoints": 2,
            "x": [-1.5, -0.75, 0.0, 0.75, 1.5],
            "y": [0, 0, 1, 2, 2],
        },
    )


def _partially_observed_mvn() -> ReferenceModelCase:
    @model
    class PartiallyObservedMvn:
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        chol = Data.matrix(n, n)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        observed_values = Data.vector(n_obs)

        y = PartiallyObserved.vector(
            MultivariateNormal(0.0, chol),
            length=n,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )

    return ReferenceModelCase(
        name="partially_observed_mvn",
        model_cls=PartiallyObservedMvn,
        meta=_meta(PartiallyObservedMvn),
        bind_values={
            "n": 3,
            "n_obs": 2,
            "n_mis": 1,
            "chol": [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.25, 0.4375, 0.85]],
            "observed_idx": [0, 2],
            "missing_idx": [1],
            "observed_values": [0.7, -0.4],
        },
    )


def _bounded_rates() -> ReferenceModelCase:
    @model
    class BoundedRates:
        p = Param(Beta(2.0, 2.0), constraint=UnitInterval())
        level = Param(
            Truncated(Normal(1.0, 1.0), lower=-1.0, upper=3.0),
            constraint=Interval(-1.0, 3.0),
        )
        y = Observed(Bernoulli(p))

    return ReferenceModelCase(
        name="bounded_rates",
        model_cls=BoundedRates,
        meta=_meta(BoundedRates),
        bind_values={"y": [0, 1, 1, 0, 1]},
    )


def _censored_exponential() -> ReferenceModelCase:
    @model
    class CensoredExponential:
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        observed_values = Data.vector(n_obs)
        missing_lower = Data.vector(n_mis)

        rate = Param(Exponential(1.0), constraint=Positive())
        y = PartiallyObserved.vector(
            Exponential(rate),
            length=n,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            missing_lower=missing_lower,
        )

    return ReferenceModelCase(
        name="censored_exponential",
        model_cls=CensoredExponential,
        meta=_meta(CensoredExponential),
        bind_values={
            "n": 6,
            "n_obs": 4,
            "n_mis": 2,
            "observed_idx": [0, 1, 3, 5],
            "missing_idx": [2, 4],
            "observed_values": [0.25, 0.9, 0.4, 1.2],
            "missing_lower": [1.5, 2.0],
        },
    )


def _vector_bounds_named_owner() -> ReferenceModelCase:
    @model
    class VectorBoundsNamedOwner:
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        observed_values = Data.vector(n_obs)
        missing_upper = Data.vector(n_mis)

        y = PartiallyObserved.vector(
            Exponential(1.0),
            length=n,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            missing_upper=missing_upper,
        )

    declared_meta = _meta(VectorBoundsNamedOwner)
    owner = declared_meta.stochastic_sites[0]
    factor = ResolvedStochasticSite(
        name="penalty",
        distribution=Normal(0.0, 1.0),
        value=owner.value,
    )
    # The eDSL has no general Factor declaration yet. Construct the adversarial
    # resolved metadata explicitly: a differently named full-vector factor comes
    # before the same-name PartiallyObserved owner and must not supply its support.
    meta = replace(declared_meta, stochastic_sites=(factor, owner))

    return ReferenceModelCase(
        name="vector_bounds_named_owner",
        model_cls=VectorBoundsNamedOwner,
        meta=meta,
        bind_values={
            "n": 2,
            "n_obs": 1,
            "n_mis": 1,
            "observed_idx": [0],
            "missing_idx": [1],
            "observed_values": [0.5],
            "missing_upper": [1.0],
        },
    )


def _interval_censored_normal() -> ReferenceModelCase:
    @model
    class IntervalCensoredNormal:
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        observed_values = Data.vector(n_obs)
        missing_lower = Data.vector(n_mis)
        missing_upper = Data.vector(n_mis)

        mu = Param(Normal(0.0, 1.0))
        y = PartiallyObserved.vector(
            Normal(mu, 1.0),
            length=n,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            missing_lower=missing_lower,
            missing_upper=missing_upper,
        )

    return ReferenceModelCase(
        name="interval_censored_normal",
        model_cls=IntervalCensoredNormal,
        meta=_meta(IntervalCensoredNormal),
        bind_values={
            "n": 5,
            "n_obs": 3,
            "n_mis": 2,
            "observed_idx": [0, 2, 4],
            "missing_idx": [1, 3],
            "observed_values": [-0.3, 0.8, 1.1],
            "missing_lower": [-1.0, 0.25],
            "missing_upper": [0.5, 1.75],
        },
    )


def reference_model_cases() -> tuple[ReferenceModelCase, ...]:
    """Return all corpus reference models in their pinned order."""
    return (
        _linear_regression(),
        _alternative_prior_regression(),
        _eight_schools_non_centered(),
        _mvn_non_centered(),
        _varying_intercepts_poisson(),
        _composed_measurements(),
        _ordinal_regression(),
        _partially_observed_mvn(),
        _bounded_rates(),
        _censored_exponential(),
        _interval_censored_normal(),
        _vector_bounds_named_owner(),
    )
