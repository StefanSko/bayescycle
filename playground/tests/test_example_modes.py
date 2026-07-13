from playwright.sync_api import Page, expect


def test_examples_disclose_only_relevant_documents(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    menu = page.locator("#examples-menu")
    expect(menu.locator("option")).to_have_count(4)
    menu.select_option("eight-schools")
    expect(page.locator("#simulation-documents")).not_to_have_attribute("open", "")
    menu.select_option("linear-simulation")
    expect(page.locator("#simulation-documents")).to_have_attribute("open", "")
