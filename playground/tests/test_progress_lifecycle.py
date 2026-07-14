from playwright.sync_api import Page, expect


def test_generation_preserves_completed_fit_progress(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#generate-button")).to_be_enabled(timeout=120_000)
    page.locator("#chains").fill("2")
    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("4")
    page.locator("#generate-button").click()
    expect(page.locator("#dataset-source-generated")).to_be_enabled(timeout=120_000)
    page.locator("#dataset-source-generated").check()
    page.locator("#fit-button").click()
    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#progress")).to_contain_text("chain 1")
    page.locator("#param-source-fixed").check()
    page.locator("#generate-button").click()
    expect(page.locator("#progress")).to_contain_text("chain 1")
