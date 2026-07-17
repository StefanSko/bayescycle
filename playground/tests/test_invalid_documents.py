from playwright.sync_api import Page, expect


def test_invalid_observed_json_is_a_visible_run_error(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    page.locator("#compile-button").click()
    expect(page.locator("#fit-button")).to_be_enabled(timeout=120_000)
    page.locator("#observed-data").fill("{")
    page.locator("#fit-button").click()
    expect(page.locator("#run-error")).to_be_visible()
    expect(page.locator("#run-error")).to_contain_text("invalid JSON")


def test_invalid_generation_json_is_a_visible_authoring_error(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#generate-button")).to_be_enabled(timeout=120_000)

    page.locator("#design-json-toggle").click()
    page.locator("#design-data").fill("{")
    expect(page.locator("#authoring-error")).to_contain_text("Design JSON: invalid JSON")
    page.locator("#design-data").fill('{"x":[-1,0,1]}')
    expect(page.locator("#authoring-error")).to_be_hidden()

    page.locator("#truth-data").fill("{")
    expect(page.locator("#authoring-error")).to_contain_text("Fixed parameter JSON: invalid JSON")
    expect(page.locator("#generate-button")).to_be_disabled()
