from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page


def test_engine_worker_message_boundary(
    page: Page, base_url: str, run_suite: Callable[..., list[dict[str, Any]]]
) -> None:
    results = run_suite(page, base_url, "engine-boundary")
    assert [result["name"] for result in results] == [
        "8 MiB plus one posterior response is typed and terminates its worker before decode",
        "posterior response at exactly 8 MiB passes the worker boundary",
        "engine worker rejects malformed and unknown responses",
        "first chain failure and run abort terminate sibling workers",
        "aborting an engine execution terminates its worker with a typed cancellation",
    ]
    assert all(result["ok"] for result in results), results
