from playwright.sync_api import Page, expect

COMPILE_TIMEOUT = 30_000
SIMULATE_TIMEOUT = 60_000
SAMPLE_TIMEOUT = 180_000


def test_eight_schools_vector_truth_simulate_and_recover(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")
    menu = page.locator("#examples-menu")
    expect(menu.locator("option")).to_have_count(14)
    menu.select_option("eight_schools_non_centered")
    page.wait_for_function(
        "window.__playground.state().irHash !== ''",
        timeout=COMPILE_TIMEOUT,
    )
    page.locator("#data-mode-design").check()

    z_row = page.locator('#truth-values tr[data-truth-name="z"]')
    expect(z_row.locator("td").first).to_contain_text("z × n_schools")
    expect(page.locator("#run-simulate")).to_be_enabled()
    page.locator("#run-simulate").click()
    page.wait_for_function(
        "window.__playground.state().simulated === true",
        timeout=SIMULATE_TIMEOUT,
    )
    for input_name in ("n_schools", "sigma", "y"):
        expect(page.locator(f'#mapping-table tr[data-input="{input_name}"]')).to_have_attribute(
            "data-status", "bound"
        )
    expect(page.locator("#run-error")).to_be_hidden()
    expect(page.locator("#run-button")).to_have_text("Sample on simulated data")

    for selector, value in {
        "#chains": "1",
        "#num-warmup": "50",
        "#num-draws": "50",
    }.items():
        page.locator(selector).fill(value)
    page.locator("#run-button").click()
    page.wait_for_function(
        "window.__playground.state().running === false && "
        "window.__playground.state().recovery !== null",
        timeout=SAMPLE_TIMEOUT,
    )

    recovery = page.evaluate("window.__playground.state().recovery")
    markers = page.locator("#plot-precis svg line.truth-marker")
    expect(markers).to_have_count(len(recovery))
    assert len(recovery) >= 4
    assert any(name.startswith("z[") for name in recovery)
    expect(page.locator("#run-error")).to_be_hidden()
