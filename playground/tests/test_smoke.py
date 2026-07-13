from collections.abc import Callable
from typing import Any

import pytest
from conftest import OriginIsolation
from playwright.sync_api import Page


@pytest.mark.parametrize("case_name", ["pyodide boots", "engine wasm compiles"])
def test_smoke_suite_case(
    page: Page,
    base_url: str,
    run_suite: Callable[[Page, str, str], list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "smoke")
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"smoke harness did not report case {case_name!r}: {results}"
    result = matching[0]
    assert result["ok"], f"{case_name}: {result['error']}"


def test_index_is_same_origin_only(
    page: Page, base_url: str, origin_isolation: OriginIsolation
) -> None:
    page.goto(f"{base_url}/site/index.html")
    assert page.title() == "Playground"
    origin_isolation.assert_only_origin()
