from playwright.sync_api import Page, expect


def test_invalid_observed_json_is_a_visible_run_error(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    page.locator("#compile-button").click()
    expect(page.locator("#sample-button")).to_be_enabled(timeout=120_000)
    page.locator("#observed-data").fill("{")
    page.locator("#sample-button").click()
    expect(page.locator("#run-error")).to_be_visible()
    expect(page.locator("#run-error")).to_contain_text("invalid JSON")
