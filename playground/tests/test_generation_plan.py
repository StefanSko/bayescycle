from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "generation convenience equals draw of joint predict for every source",
    "serialized fixed plan matches shared versioned fixture and identities",
    "rejects designSource as a generation option",
    "serializes design-source shared fixture and rejects malformed provenance",
    "serializes exact model-prior and posterior source variants",
    "plans own bytes and reject functions DOM and backend-like values",
    "rejects plan bounds unknown fields and mutable executable shapes",
]


def test_generation_plan_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "generation-plan")
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
