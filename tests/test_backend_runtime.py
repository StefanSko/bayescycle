from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from bayescycle._backend_runtime import (
    BackendRuntimeOptions,
    resolve_prior_predictive_backend,
    resolve_recover_backend,
    resolve_sample_backend,
    resolve_sbc_backend,
    resolve_simulate_backend,
)
from bayescycle._errors import WorkflowError
from bayescycle.backends.bayesite import BayesiteBackend
from bayescycle.backends.jaxstanv5 import Jaxstanv5Backend


def _write_fake_bayesite(tmp_path: Path, *commands: str) -> Path:
    engine = tmp_path / "fake_bayesite.py"
    usage = "\n".join(f"usage: bayesite {command}" for command in commands)
    engine.write_text(
        f"#!{sys.executable}\nimport sys\nprint({usage!r})\nraise SystemExit(0)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)
    return engine


def test_sample_backend_runtime_preflights_and_builds_bayesite_backend(
    tmp_path: Path,
) -> None:
    engine = _write_fake_bayesite(tmp_path, "sample")

    backend = resolve_sample_backend(
        BackendRuntimeOptions(
            backend="bayesite",
            engine=str(engine),
            extra_args=("--experimental",),
            preflight=True,
        )
    )

    assert isinstance(backend, BayesiteBackend)
    assert backend.engine == str(engine.resolve())
    assert backend.extra_args == ("--experimental",)


def test_sample_backend_runtime_skips_bayesite_preflight_for_plan_only(
    tmp_path: Path,
) -> None:
    missing_engine = tmp_path / "missing-bayesite"

    backend = resolve_sample_backend(
        BackendRuntimeOptions(
            backend="bayesite",
            engine=str(missing_engine),
            extra_args=(),
            preflight=False,
        )
    )

    assert isinstance(backend, BayesiteBackend)
    assert backend.engine == str(missing_engine)


def test_sample_backend_runtime_rejects_bayesite_options_for_jaxstanv5() -> None:
    with pytest.raises(WorkflowError, match="--engine configures the bayesite backend"):
        resolve_sample_backend(
            BackendRuntimeOptions(
                backend="jaxstanv5",
                engine="/tmp/bayesite",
                extra_args=(),
                preflight=False,
            )
        )

    with pytest.raises(WorkflowError, match="passthrough"):
        resolve_sample_backend(
            BackendRuntimeOptions(
                backend="jaxstanv5",
                engine=None,
                extra_args=("--debug",),
                preflight=False,
            )
        )


def test_prior_predictive_backend_runtime_resolves_jaxstanv5_backend() -> None:
    backend = resolve_prior_predictive_backend(
        BackendRuntimeOptions(
            backend="jaxstanv5",
            engine=None,
            extra_args=(),
            preflight=False,
        )
    )

    assert isinstance(backend, Jaxstanv5Backend)


@pytest.mark.parametrize(
    ("resolver", "command"),
    (
        (resolve_simulate_backend, "simulate"),
        (resolve_recover_backend, "recover"),
        (resolve_sbc_backend, "sbc"),
    ),
)
def test_model_command_runtime_preflights_and_builds_bayesite_backend(
    tmp_path: Path,
    resolver: Callable[[BackendRuntimeOptions], object],
    command: str,
) -> None:
    engine = _write_fake_bayesite(tmp_path, command)

    backend = resolver(
        BackendRuntimeOptions(
            backend="bayesite",
            engine=str(engine),
            extra_args=("--experimental",),
            preflight=True,
        )
    )

    assert isinstance(backend, BayesiteBackend)
    assert backend.engine == str(engine.resolve())
    assert backend.extra_args == ("--experimental",)


def test_simulate_backend_runtime_rejects_unsupported_backend() -> None:
    with pytest.raises(
        WorkflowError,
        match="backend jaxstanv5 does not support bayescycle capability simulate",
    ):
        resolve_simulate_backend(
            BackendRuntimeOptions(
                backend="jaxstanv5",
                engine=None,
                extra_args=(),
                preflight=False,
            )
        )
