"""Typed CLI settings normalized at workflow boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from bayescycle._errors import WorkflowError

DEFAULT_SEED = 0
DEFAULT_CHAINS = 4
DEFAULT_WARMUP = 1000
DEFAULT_DRAWS = 1000
DEFAULT_MAX_TREE_DEPTH = 10
DEFAULT_TARGET_ACCEPT = 0.8
MAX_REPORTABLE_I64 = 9_223_372_036_854_775_807


@dataclass(frozen=True)
class ResolvedSamplerSettings:
    """Typed sampler settings used by the in-process backend."""

    seed: int
    chains: int
    warmup: int
    draws: int
    max_tree_depth: int
    target_accept: float

    def as_json(self) -> dict[str, int | float]:
        """Return a JSON-ready settings summary."""
        return {
            "seed": self.seed,
            "chains": self.chains,
            "warmup": self.warmup,
            "draws": self.draws,
            "max_tree_depth": self.max_tree_depth,
            "target_accept": self.target_accept,
        }


@dataclass(frozen=True)
class ResolvedPriorPredictiveSettings:
    """Typed prior-predictive settings used by the in-process backend."""

    seed: int
    draws: int

    def as_json(self) -> dict[str, int]:
        """Return a JSON-ready settings summary."""
        return {"seed": self.seed, "draws": self.draws}


@dataclass(frozen=True)
class PriorPredictiveSettings:
    """Prior-predictive CLI settings forwarded or resolved by backend."""

    seed: str | None
    draws: str | None

    def resolve_for_in_process(self) -> ResolvedPriorPredictiveSettings:
        """Parse prior-predictive settings for direct Python execution."""
        return ResolvedPriorPredictiveSettings(
            seed=parse_nonnegative_int(self.seed, "--seed", default=DEFAULT_SEED),
            draws=parse_positive_int(self.draws, "--draws", default=DEFAULT_DRAWS),
        )


@dataclass(frozen=True)
class SamplerSettings:
    """Common sampler CLI settings forwarded or resolved by backend."""

    seed: str | None
    chains: str | None
    warmup: str | None
    draws: str | None
    max_tree_depth: str | None
    target_accept: str | None

    def resolve_for_in_process(self) -> ResolvedSamplerSettings:
        """Parse sampler settings for direct Python execution."""
        seed = parse_nonnegative_int(self.seed, "--seed", default=DEFAULT_SEED)
        chains = parse_positive_int(self.chains, "--chains", default=DEFAULT_CHAINS)
        warmup = parse_positive_int(self.warmup, "--warmup", default=DEFAULT_WARMUP)
        draws = parse_positive_int(self.draws, "--draws", default=DEFAULT_DRAWS)
        if draws < 4:
            raise WorkflowError(
                "--draws must be at least 4 because posterior artifacts include diagnostics"
            )
        max_tree_depth = parse_positive_int(
            self.max_tree_depth, "--max-treedepth", default=DEFAULT_MAX_TREE_DEPTH
        )
        if max_tree_depth > 20:
            raise WorkflowError("--max-treedepth must be in 1..=20")
        target_accept = parse_probability(
            self.target_accept, "--target-accept", default=DEFAULT_TARGET_ACCEPT
        )
        return ResolvedSamplerSettings(
            seed=seed,
            chains=chains,
            warmup=warmup,
            draws=draws,
            max_tree_depth=max_tree_depth,
            target_accept=target_accept,
        )


def parse_nonnegative_int(value: str | None, name: str, *, default: int) -> int:
    """Parse a non-negative artifact-reportable integer setting."""
    parsed = parse_int(value, name, default=default)
    if parsed < 0:
        raise WorkflowError(f"{name} must be non-negative")
    if parsed > MAX_REPORTABLE_I64:
        raise WorkflowError(
            f"{name} must be in 0..={MAX_REPORTABLE_I64} because artifacts report it "
            "as a JSON integer"
        )
    return parsed


def parse_positive_int(value: str | None, name: str, *, default: int) -> int:
    """Parse a positive artifact-reportable integer setting."""
    parsed = parse_int(value, name, default=default)
    if parsed < 1:
        raise WorkflowError(f"{name} must be at least 1")
    if parsed > MAX_REPORTABLE_I64:
        raise WorkflowError(
            f"{name} must be in 1..={MAX_REPORTABLE_I64} because artifacts report it "
            "as a JSON integer"
        )
    return parsed


def parse_int(value: str | None, name: str, *, default: int) -> int:
    """Parse an integer setting or return its default."""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise WorkflowError(f"{name} must be an integer") from exc


def parse_probability(value: str | None, name: str, *, default: float) -> float:
    """Parse an open-unit-interval probability setting."""
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise WorkflowError(f"{name} must be a number in (0, 1)") from exc
    if not 0.0 < parsed < 1.0:
        raise WorkflowError(f"{name} must be in (0, 1)")
    return parsed
