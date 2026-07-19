from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "fixed and model-prior plans lower to one exact native request",
    "runtime publishes and verifies composed-prior authored provenance",
    "runtime rejects caller asserted posterior association",
    "runtime association uses the fit identity captured during snapshotting",
    "runtime snapshots structural generation plan bytes before reuse",
    "publication guard and portable validation enforce the 8 MiB source cap",
    "generation abort after engine output rejects before publication",
    "runtime rejects malformed successful generation output",
    "runtime generates paired datasets through the vendored wasm",
    "runtime returns named diagnostic artifacts",
    "typed diagnostic cancellation rejects conditioning",
    "typed recovery cancellation rejects conditioning",
    "runtime propagates abort signals to diagnostic and recovery verbs",
    "retired legacy operations are rejected",
    "runtime preserves exact model bytes and returns enveloped run inputs",
    "selected canonical data bytes remain byte-exact through conditioning",
    "runtime serializes accepted object data into the sample artifact",
    "runtime conditions on one dataset through the workflow operation",
    "runtime accepts sampling settings at documented boundaries",
    "runtime posterior limit option is bounded by 64 MiB",
    "runtime accepts a posterior above the generation input limit end to end",
    "runtime rejects aggregate posterior bytes above 64 MiB before decode",
    "runtime rejects out-of-range sampling settings before executor dispatch",
    "first chain failure aborts every sibling execution",
    "run abort propagates to every in-flight chain",
    "runtime sampling returns valid merged posterior and progress",
]


def test_runtime_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "runtime", 120_000)
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
