from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "all 11 corpus models match native hashes and golden IR",
    "scenario compile composes one partial prior with the current model",
    "scenario compile accepts hierarchical support parameters",
    "scenario compile rejects source-declared data slots",
    "scenario compile rejects declared parameter dimension mismatches in every target order",
    "scenario compile attributes shared-coordinate conflicts to authored support parameters",
    "scenario compile rejects unused prior parameters on the complete path",
    "scenario compile reports prior-only model counts and unknown names",
    "scenario compile retains target outcome dimensions",
    "scenario compile rejects forbidden prior-only factors verbatim",
    "scenario compile retains target non-parameter free values",
    "surfaces bayeswire declaration errors verbatim",
    "requires exactly one model class",
    "index vector design slots are marked integer",
    "selects the sole unreferenced composed root",
    "selects a with_prior result over its source and target",
    "rejects a dependency cycle for one local model",
    "compiler executes in a dedicated worker",
    "module poisoning cannot affect the next compiler worker",
    "compiler timeout resets the isolated worker",
]


def test_compile_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "compile", 300_000)
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
