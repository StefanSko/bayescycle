from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "fixed and model-prior plans lower to one exact native request",
    "runtime rejects caller asserted posterior association",
    "runtime rejects malformed successful generation output",
    "runtime generates paired datasets through the vendored wasm",
    "runtime returns named diagnostic artifacts",
    "retired legacy operations are rejected",
    "runtime preserves exact model bytes and returns enveloped run inputs",
    "runtime serializes accepted object data into the sample artifact",
    "runtime conditions on one dataset through the workflow operation",
    "runtime sampling returns valid merged posterior and progress",
]


def test_runtime_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "runtime", 120_000)
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
