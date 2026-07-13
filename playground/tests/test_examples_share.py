from playwright.sync_api import Page, expect


def test_example_and_share_require_explicit_compile(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    expect(page.locator("#examples-menu")).to_be_attached()
    page.locator("#examples-menu").select_option("eight-schools")
    expect(page.locator("#model-source")).not_to_have_value("")
    expect(page.locator("#observed-data")).not_to_have_value("")
    expect(page.locator("#ir-hash")).to_be_hidden()
    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)

    page.locator("#share-button").click()
    expect(page.locator("#share-url")).to_be_visible()
    shared_url = page.locator("#share-url").input_value()

    fresh = page.context.browser.new_page()
    try:
        fresh.goto(shared_url)
        expect(fresh.locator("#share-review")).to_be_visible()
        expect(fresh.locator("#share-source")).to_contain_text("EightSchools")
        expect(fresh.locator("#ir-hash")).to_be_hidden()
        fresh.locator("#load-shared").click()
        expect(fresh.locator("#model-source")).to_contain_text("EightSchools")
        expect(fresh.locator("#ir-hash")).to_be_hidden()
        fresh.locator("#compile-button").click()
        expect(fresh.locator("#ir-hash")).to_be_visible(timeout=120_000)
    finally:
        fresh.close()
