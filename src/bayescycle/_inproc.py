"""In-process jaxstanv5 sampling backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from jaxstanv5.model.bound import BoundModel

from bayescycle._commands import Jaxstanv5PriorPredictiveCommand, Jaxstanv5SampleCommand
from bayescycle.data import DataDocError, data_doc_to_plain_json, read_data_doc


class InProcessBackendError(RuntimeError):
    """Raised when the in-process backend cannot complete a run."""


class _BindableModel(Protocol):
    def bind(self, **values: object) -> BoundModel: ...


class _PriorPredictiveFunction(Protocol):
    def __call__(
        self,
        model_cls: object,
        *,
        seed: int,
        num_samples: int,
        data: dict[str, object] | None = None,
    ) -> object: ...


class _SampleFunction(Protocol):
    def __call__(
        self,
        bound: BoundModel,
        seed: int,
        num_warmup: int,
        num_samples: int,
        *,
        num_chains: int,
        target_acceptance_rate: float,
        max_tree_depth: int,
    ) -> object: ...


def run_jaxstanv5_sample(command: Jaxstanv5SampleCommand) -> int:
    """Run jaxstanv5 sampling in this Python process and write posterior.ndjson."""
    sample = _load_sample_function()
    try:
        from bayescycle._posterior_ndjson import (
            PosteriorArtifactError,
            PosteriorArtifactSettings,
            _SamplerResult,
            model_data_fingerprint,
            write_posterior_ndjson,
        )
    except ImportError as exc:
        raise InProcessBackendError(
            "jaxstanv5 in-process backend requires JAX and BlackJAX. "
            "Install with `bayescycle[inproc]` or use `--backend bayesite`."
        ) from exc
    data = _load_data(command.data_path.path)
    try:
        model_cls = cast(_BindableModel, command.loaded_model.model_cls)
        bound = model_cls.bind(**data)
        result = sample(
            bound,
            command.settings.seed,
            command.settings.warmup,
            command.settings.draws,
            num_chains=command.settings.chains,
            target_acceptance_rate=command.settings.target_accept,
            max_tree_depth=command.settings.max_tree_depth,
        )
        fingerprint = model_data_fingerprint(
            command.ir_path.path.read_bytes(), command.data_path.path.read_bytes()
        )
        write_posterior_ndjson(
            command.draws_path,
            bound=bound,
            result=cast(_SamplerResult, result),
            settings=PosteriorArtifactSettings(
                seed=command.settings.seed,
                chains=command.settings.chains,
                warmup=command.settings.warmup,
                draws=command.settings.draws,
                max_tree_depth=command.settings.max_tree_depth,
                target_accept=command.settings.target_accept,
            ),
            fingerprint=fingerprint,
        )
    except (PosteriorArtifactError, TypeError, ValueError) as exc:
        raise InProcessBackendError(str(exc)) from exc
    return 0


def run_jaxstanv5_prior_predictive(command: Jaxstanv5PriorPredictiveCommand) -> int:
    """Run jaxstanv5 prior-predictive simulation and write prior_predictive.ndjson."""
    simulate_prior_predictive = _load_prior_predictive_function()
    try:
        from bayescycle._prior_predictive_ndjson import (
            PriorPredictiveArtifactError,
            _PriorPredictiveResult,
            write_prior_predictive_ndjson,
        )
    except ImportError as exc:
        raise InProcessBackendError(
            "jaxstanv5 in-process backend requires JAX. "
            "Install with `bayescycle[inproc]` or use `--backend bayesite`."
        ) from exc
    data = _load_data(command.data_path.path)
    try:
        result = simulate_prior_predictive(
            command.loaded_model.model_cls,
            seed=command.settings.seed,
            num_samples=command.settings.draws,
            data=data,
        )
        write_prior_predictive_ndjson(
            command.output_path,
            result=cast(_PriorPredictiveResult, result),
            settings=command.settings,
        )
    except (PriorPredictiveArtifactError, TypeError, ValueError) as exc:
        raise InProcessBackendError(str(exc)) from exc
    return 0


# Backward-compatible name while callers migrate to backend capability methods.
run_in_process_sample = run_jaxstanv5_sample


def _load_prior_predictive_function() -> _PriorPredictiveFunction:
    try:
        from jaxstanv5.simulation import simulate_prior_predictive
    except ImportError as exc:
        raise InProcessBackendError(
            "jaxstanv5 in-process backend requires JAX. "
            "Install with `bayescycle[inproc]` or use `--backend bayesite`."
        ) from exc
    return cast(_PriorPredictiveFunction, simulate_prior_predictive)


def _load_sample_function() -> _SampleFunction:
    try:
        from jaxstanv5.inference import sample
    except ImportError as exc:
        raise InProcessBackendError(
            "jaxstanv5 in-process backend requires JAX and BlackJAX. "
            "Install with `bayescycle[inproc]` or use `--backend bayesite`."
        ) from exc
    return cast(_SampleFunction, sample)


def _load_data(path: Path) -> dict[str, object]:
    try:
        return cast(dict[str, object], data_doc_to_plain_json(read_data_doc(path)))
    except DataDocError as exc:
        raise InProcessBackendError(f"data.json is not a valid data artifact: {exc}") from exc
