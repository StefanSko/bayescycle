from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"


def test_malformed_compiler_ir_cannot_become_successful_inference(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    source = (FIXTURES / "corpus" / "eight_schools_non_centered.py").read_text()
    page.locator("#model-source").fill(
        'import bayeswire.ir\nbayeswire.ir.canonical_bytes = lambda _meta: b"{}"\n' + source
    )
    page.locator("#observed-data").fill(
        (FIXTURES / "engine" / "eight_schools_non_centered" / "data.json").read_text()
    )
    page.locator("#compile-button").click()
    expect(page.locator("#sample-button")).to_be_enabled(timeout=120_000)
    page.locator("#chains").fill("1")
    page.locator("#warmup").fill("0")
    page.locator("#draws").fill("4")
    page.locator("#sample-button").click()
    expect(page.locator("#run-error")).to_be_visible(timeout=120_000)
    expect(page.locator("#run-error")).to_contain_text("UnsupportedIRVersion")
    expect(page.locator("#artifact-posterior")).to_be_hidden()
