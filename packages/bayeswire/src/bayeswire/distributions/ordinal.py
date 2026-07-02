"""Ordinal distribution metadata."""

from __future__ import annotations

from dataclasses import dataclass

from bayeswire.distributions.core import DiscreteDistribution, DistributionParameter


@dataclass(frozen=True)
class OrderedLogistic(DiscreteDistribution):
    """Ordered-logistic distribution metadata with zero-based category labels."""

    eta: DistributionParameter
    cutpoints: DistributionParameter
