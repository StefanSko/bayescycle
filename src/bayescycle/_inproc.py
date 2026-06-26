"""In-process jaxstanv5 sampling backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast

from jaxstanv5.model.bound import BoundModel

from bayescycle._commands import Jaxstanv5SampleCommand


class InProcessBackendError(RuntimeError):
    """Raised when the in-process backend cannot complete a run."""


class _BindableModel(Protocol):
    def bind(self, **values: object) -> BoundModel: ...


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
    command.draws_path.unlink(missing_ok=True)
    data = _load_data(command.data_path)
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
            command.ir_path.read_bytes(), command.data_path.read_bytes()
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


# Backward-compatible name while callers migrate to backend capability methods.
run_in_process_sample = run_jaxstanv5_sample


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
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
    except json.JSONDecodeError as exc:
        raise InProcessBackendError(f"data.json is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise InProcessBackendError("data.json must contain a JSON object")
    return value
