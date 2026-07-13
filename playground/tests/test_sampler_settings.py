from playwright.sync_api import Page, expect


def test_predictive_controls_require_a_safe_seed(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#simulate-button")).to_be_enabled(timeout=120_000)
    expect(page.locator("#prior-button")).to_be_enabled()
    page.locator("#seed").fill("9007199254740993")
    expect(page.locator("#simulate-button")).to_be_disabled()
    expect(page.locator("#prior-button")).to_be_disabled()


def test_sample_controls_require_engine_minimum_draws(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    page.locator("#compile-button").click()
    expect(page.locator("#sample-button")).to_be_enabled(timeout=120_000)
    expect(page.locator("#draws")).to_have_attribute("min", "4")
    page.locator("#draws").fill("3")
    expect(page.locator("#sample-button")).to_be_disabled()
    page.locator("#draws").fill("4")
    expect(page.locator("#sample-button")).to_be_enabled()
    page.locator("#draws").fill("4.9")
    expect(page.locator("#sample-button")).to_be_disabled()
    page.locator("#draws").fill("4")
    page.locator("#seed").fill("1.9")
    expect(page.locator("#sample-button")).to_be_disabled()
    page.locator("#seed").fill("9007199254740993")
    expect(page.locator("#sample-button")).to_be_disabled()
