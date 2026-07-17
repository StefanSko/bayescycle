from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

RENDER_CASES = [
    "readDashboardData shapes the fixture",
    "stats primitives",
    "every renderer is deterministic",
    "trank grid draws one panel per parameter",
    "ess×r-hat scatter has the verdict line",
    "precis shows mean and 89% interval",
    "precis labels its value axis and recovery truth",
    "divergence verdict thresholds",
    "density overlay renders observed vs replicates",
    "prior→posterior overlay labels the overlap",
]


def test_render_suite(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "render", 120_000)
    assert [result["name"] for result in results] == RENDER_CASES
    assert all(result["ok"] for result in results), results
