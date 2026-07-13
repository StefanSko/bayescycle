from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"
COMPILE_TIMEOUT = 30_000
SIMULATE_TIMEOUT = 60_000


def test_ordinal_ordered_truth_simulates(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")
    source = (FIXTURES / "corpus" / "ordinal_regression.py").read_text()
    page.evaluate("source => window.__playground.setSource(source)", source)
    page.wait_for_function(
        "window.__playground.state().irHash !== ''",
        timeout=COMPILE_TIMEOUT,
    )
    page.locator("#data-mode-design").check()

    scalar = page.locator(
        '#design-values tr[data-design-name="n_cutpoints"] [data-design-field="value"]'
    )
    expect(scalar).to_have_value("2")
    expect(page.locator('#truth-values tr[data-truth-name="cutpoints"] td').first).to_contain_text(
        "ordered around v"
    )
    expect(page.locator("#run-simulate")).to_be_enabled()
    page.locator("#run-simulate").click()
    page.wait_for_function(
        "window.__playground.state().simulated === true",
        timeout=SIMULATE_TIMEOUT,
    )
    for input_name in ("n_cutpoints", "x", "y"):
        expect(page.locator(f'#mapping-table tr[data-input="{input_name}"]')).to_have_attribute(
            "data-status", "bound"
        )
    expect(page.locator("#run-error")).to_be_hidden()
