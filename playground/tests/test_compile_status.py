from playwright.sync_api import Page, expect


def test_successful_compile_is_announced_without_relying_on_hash(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    page.locator("#compile-button").click()
    expect(page.locator("#compile-status")).to_have_text(
        "Model compiled successfully.", timeout=120_000
    )
    expect(page.locator("#ir-hash")).to_be_visible()
