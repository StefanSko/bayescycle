"""Integration tests for lower-censored PartiallyObserved Exponential sites."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from _reference_models import censored_exponential_fixture
from _validation import summarize_scalar_draws
from bayeswire import Data, PartiallyObserved, model
from bayeswire.distributions import Exponential, MultivariateNormal

from bayesjax.inference import sample
from bayesjax.simulation import simulate_prior_predictive


def _gamma_mean(shape: float, rate: float) -> float:
    return shape / rate


def _gamma_variance(shape: float, rate: float) -> float:
    return shape / (rate * rate)


def _mcse_variance(draws: jax.Array, *, ess: float) -> float:
    flat = jnp.ravel(draws)
    variance = jnp.var(flat, ddof=1)
    centered = flat - jnp.mean(flat)
    fourth = jnp.mean(centered**4)
    asymptotic_variance = jnp.maximum(fourth - variance**2, 0.0)
    return float(jnp.sqrt(asymptotic_variance / ess))


def test_censored_exponential_rate_matches_conjugate_gamma_posterior() -> None:
    fixture = censored_exponential_fixture()
    result = sample(
        fixture.bound,
        seed=1_701,
        num_warmup=500,
        num_samples=1_000,
        num_chains=4,
        target_acceptance_rate=0.9,
    )

    rate_summary = summarize_scalar_draws(result.samples, parameter=fixture.parameter)
    rate_draws = result.samples[fixture.parameter]
    sample_variance = float(jnp.var(jnp.ravel(rate_draws), ddof=1))
    reference_mean = _gamma_mean(fixture.posterior_shape, fixture.posterior_rate)
    reference_variance = _gamma_variance(fixture.posterior_shape, fixture.posterior_rate)
    variance_mcse = _mcse_variance(rate_draws, ess=rate_summary.ess)

    assert abs(rate_summary.mean - reference_mean) <= 5.0 * rate_summary.mcse_mean
    assert abs(sample_variance - reference_variance) <= 6.0 * variance_mcse
    assert int(jnp.sum(result.diagnostics.sampling.is_divergent)) == 0
    assert jnp.all(result.samples["y"] > fixture.missing_lower)


def test_censored_model_recovers_rate_better_than_complete_case_fit() -> None:
    fixture = censored_exponential_fixture()
    censored = sample(
        fixture.bound,
        seed=1_811,
        num_warmup=500,
        num_samples=1_000,
        num_chains=4,
        target_acceptance_rate=0.9,
    )
    complete_case = sample(
        fixture.complete_case_bound,
        seed=1_811,
        num_warmup=500,
        num_samples=1_000,
        num_chains=4,
        target_acceptance_rate=0.9,
    )

    censored_mean = summarize_scalar_draws(censored.samples, parameter=fixture.parameter).mean
    complete_case_mean = summarize_scalar_draws(
        complete_case.samples,
        parameter=fixture.parameter,
    ).mean

    assert abs(censored_mean - fixture.true_rate) < 0.45
    assert complete_case_mean > fixture.true_rate * 2.0
    assert abs(censored_mean - fixture.true_rate) < abs(complete_case_mean - fixture.true_rate)
    assert int(jnp.sum(censored.diagnostics.sampling.is_divergent)) == 0
    assert int(jnp.sum(complete_case.diagnostics.sampling.is_divergent)) == 0


@model
class PriorPredictiveCensoredExponential:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    missing_lower = Data.vector(n_mis)
    y = PartiallyObserved.vector(
        Exponential(2.0),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        missing_lower=missing_lower,
    )


@model
class PriorPredictiveMvnPartiallyObserved:
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


@model
class PriorPredictiveBoundedMvnPartiallyObserved:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    chol = Data.matrix(n, n)
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    missing_lower = Data.vector(n_mis)
    y = PartiallyObserved.vector(
        MultivariateNormal(0.0, chol),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        missing_lower=missing_lower,
    )


def test_prior_predictive_censored_exponential_respects_missing_bounds() -> None:
    result = simulate_prior_predictive(
        PriorPredictiveCensoredExponential,
        seed=2_001,
        num_samples=20_000,
        data={
            "n": 4,
            "n_obs": 2,
            "n_mis": 2,
            "observed_idx": jnp.asarray([0, 2]),
            "missing_idx": jnp.asarray([1, 3]),
            "observed_values": jnp.asarray([50.0, 60.0]),
            "missing_lower": jnp.asarray([0.5, 1.25]),
        },
    )

    draws = result.observed["y"]
    unbounded = draws[:, jnp.asarray([0, 2])]
    bounded = draws[:, jnp.asarray([1, 3])]
    bounded_mean = jnp.mean(bounded, axis=0)

    assert draws.shape == (20_000, 4)
    assert jnp.all(bounded > jnp.asarray([0.5, 1.25]))
    assert jnp.all(jnp.abs(jnp.mean(unbounded, axis=0) - 0.5) < 0.025)
    assert jnp.all(jnp.abs(bounded_mean - jnp.asarray([1.0, 1.75])) < 0.03)


def test_prior_predictive_unbounded_partially_observed_mvn_draws_full_vector() -> None:
    covariance = jnp.asarray([[1.0, 0.35], [0.35, 1.4]])
    result = simulate_prior_predictive(
        PriorPredictiveMvnPartiallyObserved,
        seed=2_101,
        num_samples=30_000,
        data={
            "n": 2,
            "n_obs": 1,
            "n_mis": 1,
            "chol": jnp.linalg.cholesky(covariance),
            "observed_idx": jnp.asarray([0]),
            "missing_idx": jnp.asarray([1]),
            "observed_values": jnp.asarray([99.0]),
        },
    )

    draws = result.observed["y"]
    empirical_mean = jnp.mean(draws, axis=0)
    empirical_covariance = jnp.cov(draws.T)

    assert draws.shape == (30_000, 2)
    assert jnp.all(jnp.abs(empirical_mean) < 0.025)
    assert jnp.all(jnp.abs(empirical_covariance - covariance) < 0.04)


def test_prior_predictive_rejects_bounded_partially_observed_mvn() -> None:
    with pytest.raises(TypeError, match="bounded MVN PartiallyObserved sites"):
        simulate_prior_predictive(
            PriorPredictiveBoundedMvnPartiallyObserved,
            seed=2_201,
            num_samples=2,
            data={
                "n": 2,
                "n_obs": 1,
                "n_mis": 1,
                "chol": jnp.eye(2),
                "observed_idx": jnp.asarray([0]),
                "missing_idx": jnp.asarray([1]),
                "observed_values": jnp.asarray([0.0]),
                "missing_lower": jnp.asarray([0.5]),
            },
        )
