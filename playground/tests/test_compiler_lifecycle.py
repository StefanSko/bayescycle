from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "runtime owns the compiler adapter",
    "each compile owns a fresh source-only worker request",
    "each scenario compile owns a fresh bounded worker request",
    "worker terminates before success and declaration failure settle",
    "worker terminates on startup failure malformed response and worker error",
    "oversized compiler output is bounded before use",
    "oversized model source is rejected before worker construction",
    "model schema cardinality and all text boundaries are enforced",
    "abort during post-response hashing rejects compilation",
    "worker terminates on compile timeout and cancellation",
]


def test_compiler_lifecycle_contract(
    page: Page, base_url: str, run_suite: Callable[..., list[dict[str, Any]]]
) -> None:
    results = run_suite(page, base_url, "compiler-lifecycle")
    assert [result["name"] for result in results] == CASES
    failures = [result for result in results if not result["ok"]]
    assert not failures, failures
