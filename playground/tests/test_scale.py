from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"
COMPILE_TIMEOUT = 30_000
RUN_TIMEOUT = 180_000
EIGHT_SCHOOLS_HASH = "6cb101cf5159bddcbe10650a87a8763054a462da0f4374b8b61a2a1a861695dc"


def fixture_text(relative_path: str) -> str:
    return (FIXTURES / relative_path).read_text()


def test_scale_predictive_plots_complete(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")
    page.evaluate(
        "source => window.__playground.setSource(source)",
        fixture_text("corpus/eight_schools_non_centered.py"),
    )
    expect(page.locator("#ir-hash-chip")).to_have_text(
        EIGHT_SCHOOLS_HASH,
        timeout=COMPILE_TIMEOUT,
    )

    page.locator("#json-input").fill(fixture_text("engine/eight_schools_non_centered/data.json"))
    page.locator("#json-load").click()
    for selector, value in {
        "#chains": "2",
        "#num-warmup": "200",
        "#num-draws": "1000",
        "#seed": "5",
    }.items():
        page.locator(selector).fill(value)

    expect(page.locator("#run-button")).to_be_enabled()
    page.locator("#run-button").click()
    page.wait_for_function(
        "window.__playground.state().running === false && "
        "window.__playground.state().lastRun !== null",
        timeout=RUN_TIMEOUT,
    )

    expect(page.locator('#plot-ppc svg[role="img"]')).to_have_count(1)
    expect(page.locator('#plot-overlay svg[role="img"]')).to_have_count(1)
    expect(page.locator("#run-error")).to_be_hidden()
    expect(page.locator("#compile-error")).to_be_hidden()
