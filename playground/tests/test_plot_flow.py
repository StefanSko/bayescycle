from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"


def test_completed_posterior_renders_core_plots(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        (FIXTURES / "corpus" / "eight_schools_non_centered.py").read_text()
    )
    page.locator("#observed-data").fill(
        (FIXTURES / "engine" / "eight_schools_non_centered" / "data.json").read_text()
    )
    page.locator("#compile-button").click()
    expect(page.locator("#fit-button")).to_be_enabled(timeout=120_000)
    page.locator("#chains").fill("2")
    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("8")
    page.locator("#fit-button").click()
    expect(page.locator("#plot-trank svg")).to_be_visible(timeout=120_000)
    expect(page.locator("#plot-ess-rhat svg")).to_be_visible()
    expect(page.locator("#plot-precis svg")).to_be_visible()
    expect(page.locator("#plot-precis .value-axis")).to_have_count(1)
    expect(page.locator("#plot-precis .truth-marker")).to_have_count(0)
