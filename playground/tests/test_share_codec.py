from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "round-trips projects byte-equal",
    "round-trips the prior snippet additively while old links remain valid",
    "rejects garbage",
    "rejects compressed payloads above the character cap",
    "encoder never emits a payload above decoder limits",
    "rejects streaming decompression above the byte cap",
    "warn threshold",
]


def test_share_codec(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "share")
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
