from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

CORE_CASES = [
    ("compile", 300_000, "all 10 corpus models match native hashes and golden IR"),
    ("compile", 300_000, "surfaces bayeswire declaration errors verbatim"),
    ("compile", 300_000, "requires exactly one model class"),
    ("engine", 120_000, "golden diagnose is byte-identical"),
    ("engine", 120_000, "streams per_draw_v2 in batches"),
    ("engine", 120_000, "sample yields one output per chain and streams draws"),
    ("engine", 120_000, "diagnose merges the two chain fits"),
    ("engine", 120_000, "malformed IR returns a typed engine error"),
    ("engine", 120_000, "exports all nine verbs"),
]


@pytest.mark.parametrize(("suite_name", "timeout_ms", "case_name"), CORE_CASES)
def test_core_suite_case(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    suite_name: str,
    timeout_ms: int,
    case_name: str,
) -> None:
    results = run_suite(page, base_url, suite_name, timeout_ms)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"{suite_name} harness did not report case {case_name!r}: {results}"
    result = matching[0]
    assert result["ok"], f"{case_name}: {result['error']}"
