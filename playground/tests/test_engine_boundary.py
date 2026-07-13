from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page


def test_engine_worker_message_boundary(
    page: Page, base_url: str, run_suite: Callable[..., list[dict[str, Any]]]
) -> None:
    results = run_suite(page, base_url, "engine-boundary")
    assert [result["name"] for result in results] == [
        "engine worker rejects malformed and unknown responses"
    ]
    assert results[0]["ok"], results
