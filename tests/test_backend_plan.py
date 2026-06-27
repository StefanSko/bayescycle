from __future__ import annotations

from pathlib import Path

import pytest

from bayescycle._backend_plan import (
    BackendPlanRequest,
    ExplicitMixedBackendPlan,
    SingleBackendPlan,
    resolve_backend_plan,
    resolve_backend_plan_file,
)
from bayescycle._errors import WorkflowError


def test_single_backend_plan_applies_to_all_stages() -> None:
    resolved = resolve_backend_plan(
        BackendPlanRequest(
            backend="bayesite",
            simulate_backend=None,
            recover_backend=None,
            engine="/tmp/bayesite",
        )
    )

    assert isinstance(resolved.plan, SingleBackendPlan)
    assert resolved.as_json() == {
        "mode": "single",
        "stages": {"simulate": "bayesite", "recover": "bayesite"},
        "backends": {"bayesite": {"engine": "/tmp/bayesite"}},
    }


def test_omitted_backend_resolves_one_default_backend() -> None:
    resolved = resolve_backend_plan(
        BackendPlanRequest(
            backend=None,
            simulate_backend=None,
            recover_backend=None,
            engine=None,
        )
    )

    assert resolved.as_json()["stages"] == {"simulate": "bayesite", "recover": "bayesite"}


def test_complete_stage_plan_can_be_mixed() -> None:
    resolved = resolve_backend_plan(
        BackendPlanRequest(
            backend=None,
            simulate_backend="bayesite",
            recover_backend="jaxstanv5",
            engine="/tmp/bayesite",
        )
    )

    assert isinstance(resolved.plan, ExplicitMixedBackendPlan)
    assert resolved.as_json()["stages"] == {"simulate": "bayesite", "recover": "jaxstanv5"}


def test_partial_stage_override_fails() -> None:
    with pytest.raises(WorkflowError, match="Partial backend assignment"):
        resolve_backend_plan(
            BackendPlanRequest(
                backend=None,
                simulate_backend="bayesite",
                recover_backend=None,
                engine=None,
            )
        )


def test_engine_does_not_select_bayesite_implicitly() -> None:
    with pytest.raises(WorkflowError, match="no bayesite backend stage was selected"):
        resolve_backend_plan(
            BackendPlanRequest(
                backend="jaxstanv5",
                simulate_backend=None,
                recover_backend=None,
                engine="/tmp/bayesite",
            )
        )


def test_backend_stage_support_is_validated() -> None:
    with pytest.raises(WorkflowError, match="does not support required stage simulate"):
        resolve_backend_plan(
            BackendPlanRequest(
                backend="jaxstanv5",
                simulate_backend=None,
                recover_backend=None,
                engine=None,
            )
        )


def test_mixed_toml_plan_resolves(tmp_path: Path) -> None:
    config = tmp_path / "workflow.toml"
    config.write_text(
        '[workflow]\nmode = "mixed"\n\n'
        '[stages.simulate]\nbackend = "bayesite"\n\n'
        '[stages.recover]\nbackend = "jaxstanv5"\n\n'
        '[backends.bayesite]\nengine = "/tmp/bayesite"\n',
        encoding="utf-8",
    )

    resolved = resolve_backend_plan_file(config)

    assert resolved.as_json() == {
        "mode": "mixed",
        "stages": {"simulate": "bayesite", "recover": "jaxstanv5"},
        "backends": {"bayesite": {"engine": "/tmp/bayesite"}},
    }


def test_partial_mixed_toml_plan_fails(tmp_path: Path) -> None:
    config = tmp_path / "workflow.toml"
    config.write_text(
        '[workflow]\nmode = "mixed"\n\n[stages.simulate]\nbackend = "bayesite"\n',
        encoding="utf-8",
    )

    with pytest.raises(WorkflowError, match="Partial backend assignment"):
        resolve_backend_plan_file(config)
