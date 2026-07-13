from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

CASES = [
    "all 10 corpus models match native hashes and golden IR",
    "surfaces bayeswire declaration errors verbatim",
    "requires exactly one model class",
    "compiler executes in a dedicated worker",
    "user source cannot replace the trusted IR serializer",
    "trusted serializer is absent from user-visible globals",
    "compiler timeout resets the isolated worker",
]


@pytest.mark.parametrize("case_name", CASES)
def test_compile_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "compile", 300_000)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"compile harness did not report {case_name!r}: {results}"
    assert matching[0]["ok"], f"{case_name}: {matching[0]['error']}"
