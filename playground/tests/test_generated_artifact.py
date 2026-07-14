from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "parses canonical generated header pairs and trailer",
    "accepts varying model-prior and posterior parameters",
    "selects exact nested bytes and preserves the download",
    "verifies hashes fixed values and design prefix",
    "resolver rejects output from the wrong generation plan",
    "rejects malformed truncated nonfinite and oversized streams",
]


def test_generated_artifact_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "generated-artifact")
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
