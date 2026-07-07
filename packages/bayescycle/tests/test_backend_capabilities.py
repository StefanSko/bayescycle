from __future__ import annotations

import pytest

from bayescycle._errors import WorkflowError
from bayescycle._workflow.capabilities import (
    BAYESITE,
    BAYESJAX,
    FIRST_PARTY_BACKENDS,
    BackendCapability,
    BackendId,
)


def test_backend_capability_is_closed_workflow_adt() -> None:
    assert tuple(capability.value for capability in BackendCapability) == (
        "sample",
        "prior-predictive",
        "simulate",
        "recover",
        "sbc",
        "diagnose",
        "posterior-predictive",
        "posterior-check",
        "recover-check",
    )


def test_backend_ids_are_open_values_resolved_by_catalog() -> None:
    custom = BackendId("custom-backend")

    assert str(custom) == "custom-backend"
    assert custom != BAYESITE
    assert custom != BAYESJAX
    assert FIRST_PARTY_BACKENDS.resolve("bayesite") == BAYESITE
    assert FIRST_PARTY_BACKENDS.resolve("bayesjax") == BAYESJAX

    with pytest.raises(WorkflowError, match="--backend must be one of: bayesite, bayesjax"):
        FIRST_PARTY_BACKENDS.resolve("missing")


def test_first_party_catalog_declares_capability_support() -> None:
    assert FIRST_PARTY_BACKENDS.choices() == ("bayesite", "bayesjax")
    assert FIRST_PARTY_BACKENDS.choices_for(BackendCapability.SIMULATE) == ("bayesite",)
    assert FIRST_PARTY_BACKENDS.choices_for(BackendCapability.SAMPLE) == (
        "bayesite",
        "bayesjax",
    )

    FIRST_PARTY_BACKENDS.require(BAYESITE, BackendCapability.SIMULATE)
    FIRST_PARTY_BACKENDS.require(BAYESJAX, BackendCapability.SAMPLE)

    with pytest.raises(
        WorkflowError,
        match="backend bayesjax does not support bayescycle capability simulate",
    ):
        FIRST_PARTY_BACKENDS.require(BAYESJAX, BackendCapability.SIMULATE)
