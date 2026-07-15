"""Prior and prior-predictive simulation internals."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from bayeswire.constraints.core import Constraint
from bayeswire.distributions.continuous import Exponential
from bayeswire.distributions.core import Distribution
from bayeswire.distributions.multivariate import MultivariateNormal
from bayeswire.model._factorization import _param_references
from bayeswire.model._structural import _structurally_equal
from bayeswire.model.decorator import (
    ModelMeta,
    ResolvedStochasticSite,
    model_meta,
    resolved_free_values,
    resolved_stochastic_sites,
)
from bayeswire.model.expr import DataRef, ParamRef, VectorScatterOp

from bayesjax._backends.jax.binding import (
    _normalize_declared_data_values,
    _resolve_param_shape,
    _resolve_vector_bounds,
    _validate_bound_distribution_parameters,
    _validate_bound_index_expressions,
)
from bayesjax._backends.jax.constraints import ResolvedVectorBounds
from bayesjax._backends.jax.distributions import (
    batch_shape,
    cdf,
    event_shape,
    icdf,
    is_inverse_cdf,
    is_sampleable,
)
from bayesjax._backends.jax.distributions import (
    sample as distribution_sample,
)
from bayesjax.compiler.core import _evaluate_distribution, _evaluate_expr
from bayesjax.simulation.domains import (
    OrderedVectorDomain,
    ScalarIntervalDomain,
    UnconstrainedDomain,
    prior_domain_for_constraint,
)


@dataclass(frozen=True)
class PriorPredictiveResult:
    """Draws from a model's prior and prior-predictive distribution."""

    parameters: Mapping[str, jax.Array]
    observed: Mapping[str, jax.Array]
    data: Mapping[str, jax.Array]


def _leading_sample_shape(
    *,
    target_shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
    event_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Return iid sample dimensions from a full target value shape."""
    suffix = batch_shape + event_shape
    if suffix == ():
        return target_shape
    if len(target_shape) < len(suffix) or target_shape[-len(suffix) :] != suffix:
        raise ValueError(
            "Target shape must end with distribution batch_shape + event_shape: "
            f"target_shape={target_shape}, batch_shape={batch_shape}, event_shape={event_shape}"
        )
    return target_shape[: -len(suffix)]


def _sample_interval_restricted(
    key: jax.Array,
    distribution: Distribution,
    domain: ScalarIntervalDomain,
    *,
    target_shape: tuple[int, ...],
) -> jax.Array:
    if event_shape(distribution) != ():
        raise TypeError("Interval-constrained prior simulation requires scalar-event distributions")
    sample_shape = _leading_sample_shape(
        target_shape=target_shape,
        batch_shape=batch_shape(distribution),
        event_shape=event_shape(distribution),
    )
    lower_probability = (
        jnp.asarray(0.0) if domain.lower is None else cdf(distribution, domain.lower)
    )
    upper_probability = (
        jnp.asarray(1.0) if domain.upper is None else cdf(distribution, domain.upper)
    )
    uniform = jax.random.uniform(
        key,
        shape=sample_shape + batch_shape(distribution),
        minval=lower_probability,
        maxval=upper_probability,
    )
    return icdf(distribution, uniform)


def _sample_ordered_vector(
    key: jax.Array,
    distribution: Distribution,
    *,
    target_shape: tuple[int, ...],
) -> jax.Array:
    """Sample an ordered vector from an iid scalar constrained-space prior."""
    if target_shape == ():
        raise ValueError("Ordered prior simulation requires vector target shape")
    if event_shape(distribution) != ():
        raise TypeError("Ordered prior simulation requires scalar-event distributions")
    if batch_shape(distribution) != ():
        raise TypeError("Ordered prior simulation requires iid scalar distributions")
    raw = distribution_sample(distribution, key, sample_shape=target_shape)
    return jnp.sort(raw, axis=-1)


def _sample_prior_value(
    key: jax.Array,
    distribution: Distribution,
    *,
    constraint: Constraint | None,
    target_shape: tuple[int, ...],
) -> jax.Array:
    """Sample one parameter value from its constrained-space prior."""
    domain = prior_domain_for_constraint(constraint)

    if isinstance(domain, UnconstrainedDomain):
        if is_sampleable(distribution):
            sample_shape = _leading_sample_shape(
                target_shape=target_shape,
                batch_shape=batch_shape(distribution),
                event_shape=event_shape(distribution),
            )
            return distribution_sample(distribution, key, sample_shape=sample_shape)
        raise TypeError(f"Unsupported prior distribution: {type(distribution).__name__}")

    if isinstance(domain, ScalarIntervalDomain):
        if is_inverse_cdf(distribution):
            return _sample_interval_restricted(
                key,
                distribution,
                domain,
                target_shape=target_shape,
            )
        raise TypeError(
            f"Unsupported interval-constrained prior distribution: {type(distribution).__name__}"
        )

    if isinstance(domain, OrderedVectorDomain):
        if is_sampleable(distribution):
            return _sample_ordered_vector(key, distribution, target_shape=target_shape)
        raise TypeError(f"Unsupported ordered prior distribution: {type(distribution).__name__}")

    raise TypeError(f"Unsupported prior domain: {type(domain).__name__}")


def _scalar_value_as_int(value: jax.Array, *, label: str) -> int:
    if value.ndim != 0:
        raise ValueError(f"{label} must be scalar")
    if not jnp.issubdtype(value.dtype, jnp.integer):
        raise TypeError(f"{label} must be integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _sample_unrestricted_distribution(
    key: jax.Array,
    distribution: Distribution,
    *,
    target_shape: tuple[int, ...],
) -> jax.Array:
    if not is_sampleable(distribution):
        raise TypeError(f"Unsupported prior distribution: {type(distribution).__name__}")
    sample_shape = _leading_sample_shape(
        target_shape=target_shape,
        batch_shape=batch_shape(distribution),
        event_shape=event_shape(distribution),
    )
    return distribution_sample(distribution, key, sample_shape=sample_shape)


def _sample_vector_bounds_restricted(
    key: jax.Array,
    distribution: Distribution,
    bounds: ResolvedVectorBounds,
    *,
    target_shape: tuple[int, ...],
) -> jax.Array:
    if event_shape(distribution) != () or batch_shape(distribution) != ():
        raise TypeError(
            "VectorBounds prior-predictive simulation requires iid scalar distributions"
        )
    if isinstance(distribution, Exponential) and bounds.lower is not None and bounds.upper is None:
        return bounds.lower + distribution_sample(distribution, key, sample_shape=target_shape)
    if not is_inverse_cdf(distribution):
        raise TypeError(
            f"Unsupported bounded PartiallyObserved prior distribution: "
            f"{type(distribution).__name__}"
        )
    return _sample_interval_restricted(
        key,
        distribution,
        ScalarIntervalDomain(lower=bounds.lower, upper=bounds.upper),
        target_shape=target_shape,
    )


def _partially_observed_target(
    site: ResolvedStochasticSite,
    values: dict[str, jax.Array],
) -> tuple[int, jax.Array]:
    if not isinstance(site.value, VectorScatterOp):
        raise TypeError(f"Non-parameter free value {site.name!r} is not a PartiallyObserved vector")
    length = _scalar_value_as_int(
        _evaluate_expr(site.value.length, values),
        label=f"PartiallyObserved site {site.name!r} length",
    )
    missing_idx = _evaluate_expr(site.value.missing_idx, values)
    return length, missing_idx


def _sample_partially_observed_site(
    key: jax.Array,
    site: ResolvedStochasticSite,
    distribution: Distribution,
    values: dict[str, jax.Array],
    vector_bounds: Mapping[str, ResolvedVectorBounds],
) -> tuple[jax.Array, jax.Array]:
    length, missing_idx = _partially_observed_target(site, values)
    target_shape = (length,)
    bounds = vector_bounds.get(site.name)

    if bounds is None:
        full_value = _sample_unrestricted_distribution(
            key,
            distribution,
            target_shape=target_shape,
        )
        return full_value, full_value[missing_idx]

    if isinstance(distribution, MultivariateNormal):
        raise TypeError(
            "bounded MVN PartiallyObserved sites are not supported by prior-predictive simulation"
        )

    full_key, missing_key = jax.random.split(key)
    full_value = _sample_unrestricted_distribution(
        full_key,
        distribution,
        target_shape=target_shape,
    )
    missing_value = _sample_vector_bounds_restricted(
        missing_key,
        distribution,
        bounds,
        target_shape=(missing_idx.shape[0],),
    )
    return full_value.at[missing_idx].set(missing_value), missing_value


def _normalize_data(meta: ModelMeta, data: Mapping[str, object] | None) -> dict[str, jax.Array]:
    return _normalize_declared_data_values(meta, data)


def _resolve_param_shapes(
    meta: ModelMeta, data: dict[str, jax.Array]
) -> dict[str, tuple[int, ...]]:
    return {name: _resolve_param_shape(param.size, data) for name, param in meta.params.items()}


def _resolve_free_value_shapes(
    meta: ModelMeta, data: dict[str, jax.Array]
) -> dict[str, tuple[int, ...]]:
    return {
        name: _resolve_param_shape(value.size, data)
        for name, value in resolved_free_values(meta).items()
    }


@dataclass(frozen=True)
class _PriorPredictiveOutcomeSite:
    """One factor plus the declaration name that receives its generated value."""

    name: str
    site: ResolvedStochasticSite
    partially_observed: bool


@dataclass(frozen=True)
class _PriorPredictiveSitePlan:
    """Declaration-backed outcome sites in their ancestral factor order."""

    outcomes: tuple[_PriorPredictiveOutcomeSite, ...]


def _claim_unique_generative_site(
    sites: tuple[ResolvedStochasticSite, ...],
    claimed: set[int],
    *,
    declaration: str,
    matches: Callable[[ResolvedStochasticSite], bool],
) -> tuple[int, ResolvedStochasticSite]:
    matching_indices = [
        index for index, site in enumerate(sites) if index not in claimed and matches(site)
    ]
    if len(matching_indices) != 1:
        raise TypeError(
            f"prior-predictive {declaration} requires exactly one matching generative "
            f"stochastic site, found {len(matching_indices)}; keep one declaration-backed "
            "site and remove additional density factors from the ancestrally simulated model"
        )
    index = matching_indices[0]
    claimed.add(index)
    return index, sites[index]


def _site_owns_non_param_free_value(site: ResolvedStochasticSite, name: str) -> bool:
    if site.name != name:
        return False
    if isinstance(site.value, ParamRef):
        return site.value.name == name
    return (
        isinstance(site.value, VectorScatterOp)
        and isinstance(site.value.missing_values, ParamRef)
        and site.value.missing_values.name == name
    )


def _order_ancestral_outcomes(
    outcomes: tuple[_PriorPredictiveOutcomeSite, ...],
    *,
    param_names: set[str],
) -> tuple[_PriorPredictiveOutcomeSite, ...]:
    """Stable-toposort declaration outcomes by free-value distribution dependencies."""
    available = set(param_names)
    pending = list(outcomes)
    ordered: list[_PriorPredictiveOutcomeSite] = []
    while pending:
        for index, outcome in enumerate(pending):
            dependencies = set(_param_references(outcome.site.distribution))
            if dependencies <= available:
                ordered.append(outcome)
                available.add(outcome.name)
                pending.pop(index)
                break
        else:
            unresolved = {
                outcome.name: sorted(set(_param_references(outcome.site.distribution)) - available)
                for outcome in pending
            }
            raise TypeError(
                "prior-predictive outcome dependencies are cyclic or unavailable; "
                f"unresolved references: {unresolved}"
            )
    return tuple(ordered)


def _prior_predictive_site_plan(meta: ModelMeta) -> _PriorPredictiveSitePlan:
    """Classify declaration sites and reject Factors before ancestral drawing."""
    sites = resolved_stochastic_sites(meta)
    claimed: set[int] = set()

    for name, param in meta.params.items():
        _claim_unique_generative_site(
            sites,
            claimed,
            declaration=f"Param {name!r}",
            matches=lambda site, name=name, param=param: (
                site.value == ParamRef(name)
                and _structurally_equal(site.distribution, param.distribution)
            ),
        )

    outcome_names: dict[int, str] = {}
    for observed in meta.observed_nodes:
        index, _site = _claim_unique_generative_site(
            sites,
            claimed,
            declaration=f"Observed {observed.name!r}",
            matches=lambda site, observed=observed: (
                site.value == DataRef(observed.name)
                and _structurally_equal(site.distribution, observed.distribution)
            ),
        )
        outcome_names[index] = observed.name

    partially_observed_indices: set[int] = set()
    for name in resolved_free_values(meta):
        if name in meta.params:
            continue
        index, site = _claim_unique_generative_site(
            sites,
            claimed,
            declaration=f"non-Param free value {name!r}",
            matches=lambda candidate, name=name: _site_owns_non_param_free_value(candidate, name),
        )
        if not isinstance(site.value, VectorScatterOp):
            raise TypeError(
                f"prior-predictive non-Param free value {name!r} is not a "
                "PartiallyObserved VectorScatter site"
            )
        partially_observed_indices.add(index)
        outcome_names[index] = name

    factor_sites = [
        f"{site.name!r} at stochastic_sites[{index}]"
        for index, site in enumerate(sites)
        if index not in claimed
    ]
    if factor_sites:
        raise TypeError(
            "prior-predictive cannot simulate additional stochastic factor sites "
            f"{factor_sites}; Factors affect density but have no ancestral draw, so use a "
            "model containing only declaration-backed generative sites"
        )

    outcomes = tuple(
        _PriorPredictiveOutcomeSite(
            name=outcome_names[index],
            site=site,
            partially_observed=index in partially_observed_indices,
        )
        for index, site in enumerate(sites)
        if index in outcome_names
    )
    return _PriorPredictiveSitePlan(
        outcomes=_order_ancestral_outcomes(
            outcomes,
            param_names=set(meta.params),
        )
    )


def _validate_observed_shapes(
    meta: ModelMeta,
    observed_shapes: Mapping[str, tuple[int, ...]] | None,
) -> dict[str, tuple[int, ...] | None]:
    raw_shapes: Mapping[str, tuple[int, ...]] = {} if observed_shapes is None else observed_shapes
    observed_names = {observed.name for observed in meta.observed_nodes}
    extra = set(raw_shapes) - observed_names
    if extra:
        raise ValueError(f"Unexpected observed shapes: {sorted(extra)}")

    result: dict[str, tuple[int, ...] | None] = {}
    for observed in meta.observed_nodes:
        shape = raw_shapes.get(observed.name)
        if shape is not None:
            for dim in shape:
                if isinstance(dim, bool):
                    raise TypeError("Observed shape dimensions must be integers, not bool")
                if dim < 0:
                    raise ValueError("Observed shape dimensions must be non-negative")
        result[observed.name] = shape
    return result


def _simulate_one(
    key: jax.Array,
    *,
    meta: ModelMeta,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
    observed_shapes: dict[str, tuple[int, ...] | None],
    vector_bounds: Mapping[str, ResolvedVectorBounds],
    outcomes: tuple[_PriorPredictiveOutcomeSite, ...],
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    keys = jax.random.split(key, len(meta.params) + len(outcomes))
    key_index = 0
    parameters: dict[str, jax.Array] = {}
    values = dict(data)

    for name, param in meta.params.items():
        distribution = _evaluate_distribution(param.distribution, values)
        value = _sample_prior_value(
            keys[key_index],
            distribution,
            constraint=param.constraint,
            target_shape=param_shapes[name],
        )
        parameters[name] = value
        values[name] = value
        key_index += 1

    observed_values: dict[str, jax.Array] = {}
    for outcome in outcomes:
        site = outcome.site
        distribution = _evaluate_distribution(site.distribution, values)
        if outcome.partially_observed:
            observed_value, free_value = _sample_partially_observed_site(
                keys[key_index],
                site,
                distribution,
                values,
                vector_bounds,
            )
            values[outcome.name] = free_value
        else:
            observed_target_shape = observed_shapes[outcome.name]
            if observed_target_shape is None:
                if not is_sampleable(distribution):
                    raise TypeError(
                        f"Unsupported prior distribution: {type(distribution).__name__}"
                    )
                observed_target_shape = batch_shape(distribution) + event_shape(distribution)
            observed_value = _sample_prior_value(
                keys[key_index],
                distribution,
                constraint=None,
                target_shape=observed_target_shape,
            )
            values[outcome.name] = observed_value
        observed_values[outcome.name] = observed_value
        key_index += 1

    return parameters, observed_values


def simulate_prior_predictive(
    model_cls: object,
    *,
    seed: int,
    num_samples: int,
    data: Mapping[str, object] | None = None,
    observed_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> PriorPredictiveResult:
    """Draw from a model's prior and prior predictive distribution."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    meta = model_meta(model_cls)
    site_plan = _prior_predictive_site_plan(meta)
    normalized_data = _normalize_data(meta, data)
    param_shapes = _resolve_param_shapes(meta, normalized_data)
    free_value_shapes = _resolve_free_value_shapes(meta, normalized_data)
    vector_bounds = _resolve_vector_bounds(meta, normalized_data, free_value_shapes)
    _validate_bound_index_expressions(meta, normalized_data, free_value_shapes)
    _validate_bound_distribution_parameters(meta, normalized_data, free_value_shapes)
    normalized_observed_shapes = _validate_observed_shapes(meta, observed_shapes)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)

    def draw_one(key: jax.Array) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        return _simulate_one(
            key,
            meta=meta,
            data=normalized_data,
            param_shapes=param_shapes,
            observed_shapes=normalized_observed_shapes,
            vector_bounds=vector_bounds,
            outcomes=site_plan.outcomes,
        )

    parameters, observed = jax.jit(jax.vmap(draw_one))(keys)
    return PriorPredictiveResult(parameters=parameters, observed=observed, data=normalized_data)
