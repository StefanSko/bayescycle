from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "golden diagnose is byte-identical",
    "streams per_draw_v2 in batches",
    "sample yields one output per chain and streams draws",
    "diagnose merges the two chain fits",
    "diagnose bounds merged fits by default before executor dispatch",
    "merged fits never retain first-chain diagnostics",
    "merged fit serialization enforces its byte ceiling",
    "malformed IR returns a typed engine error",
    "exports the six retained verbs including native generation",
]


def test_engine_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "engine", 120_000)
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
