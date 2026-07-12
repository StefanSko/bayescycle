from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser, Page, expect

FIXTURES = Path(__file__).parent / "fixtures"
COMPILE_TIMEOUT = 30_000
EIGHT_SCHOOLS_HASH = "6cb101cf5159bddcbe10650a87a8763054a462da0f4374b8b61a2a1a861695dc"


def fixture_text(relative_path: str) -> str:
    return (FIXTURES / relative_path).read_text()


def open_app(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")


def set_source(page: Page, source: str) -> None:
    page.evaluate("source => window.__playground.setSource(source)", source)


def author_share_url(page: Page, base_url: str) -> tuple[str, str]:
    open_app(page, base_url)
    source = fixture_text("corpus/eight_schools_non_centered.py")
    set_source(page, source)
    expect(page.locator("#ir-hash-chip")).to_have_text(
        EIGHT_SCHOOLS_HASH,
        timeout=COMPILE_TIMEOUT,
    )
    share_button = page.locator("#share-button")
    expect(share_button).to_be_enabled()
    share_button.click()
    share_url = page.locator("#share-url").input_value()
    assert share_url.startswith(f"{base_url}/site/index.html#project=")
    assert page.url == share_url
    return share_url, source


def test_share_and_load(page: Page, browser: Browser, base_url: str) -> None:
    share_url, source = author_share_url(page, base_url)
    original_hash = page.locator("#ir-hash-chip").text_content()
    assert original_hash == EIGHT_SCHOOLS_HASH

    context = browser.new_context()
    shared_page = context.new_page()
    try:
        shared_page.goto(share_url)
        shared_page.wait_for_function("window.__playground !== undefined")
        interstitial = shared_page.locator("#share-interstitial")
        expect(interstitial).to_be_visible()
        expect(interstitial).to_contain_text("this link contains model code; review before running")
        expect(shared_page.locator("#share-source")).to_contain_text("@model")
        assert shared_page.evaluate("window.__playground.getSource()") == ""
        state = shared_page.evaluate("window.__playground.state()")
        assert state.get("irHash", "") == ""
        expect(shared_page.locator("#ir-hash-chip")).to_have_text("")

        shared_page.locator("#share-load").click()
        expect(interstitial).to_be_hidden()
        assert shared_page.evaluate("window.__playground.getSource()") == source
        expect(shared_page.locator("#ir-hash-chip")).to_have_text(
            original_hash,
            timeout=COMPILE_TIMEOUT,
        )
    finally:
        context.close()


def test_never_auto_executes(page: Page, browser: Browser, base_url: str) -> None:
    share_url, _ = author_share_url(page, base_url)

    context = browser.new_context()
    shared_page = context.new_page()
    try:
        shared_page.goto(share_url)
        shared_page.wait_for_function("window.__playground !== undefined")
        expect(shared_page.locator("#share-interstitial")).to_be_visible()
        shared_page.wait_for_timeout(3_000)
        expect(shared_page.locator("#ir-hash-chip")).to_have_text("")
        state = shared_page.evaluate("window.__playground.state()")
        assert state.get("irHash", "") == ""
        assert shared_page.evaluate("window.__playground.getSource()") == ""
    finally:
        context.close()


def fixed_noise(length: int) -> str:
    state = 0x5EED1234
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@$%^&*()"
    result: list[str] = []
    for _ in range(length):
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        result.append(alphabet[state % len(alphabet)])
    return "".join(result)


def test_oversize_warns(page: Page, base_url: str) -> None:
    open_app(page, base_url)
    source = fixture_text("corpus/eight_schools_non_centered.py") + f"\n# {fixed_noise(20_000)}\n"
    set_source(page, source)
    page.locator("#json-input").fill(fixture_text("engine/eight_schools_non_centered/data.json"))
    share_button = page.locator("#share-button")
    expect(share_button).to_be_enabled()
    share_button.click()

    expect(page.locator("#share-warning")).to_be_visible()
    payload = page.url.split("#project=", maxsplit=1)[1]
    decoded = page.evaluate(
        """async payload => {
          const { decodeProject, encodeProject } = await import('/site/src/app/share.mjs');
          const project = await decodeProject(payload);
          return { project, encoded: await encodeProject(project) };
        }""",
        payload,
    )
    assert decoded["project"]["source"] == source
    assert decoded["encoded"] == payload
    assert set(decoded["project"]) <= {"v", "source", "dataMode", "design", "truth", "sampler"}


def test_examples_menu(page: Page, base_url: str) -> None:
    open_app(page, base_url)
    menu = page.locator("#examples-menu")
    expect(menu.locator("option")).to_have_count(14)
    expect(menu.locator("option").first).to_have_text("Examples…")

    menu.select_option("eight_schools_non_centered")
    page.wait_for_function("window.__playground.getSource().includes('@model')")
    assert "@model" in page.evaluate("window.__playground.getSource()")
    assert "bayescycle.data.json.v1" in page.locator("#json-input").input_value()

    menu.select_option("divorce_generative")
    page.wait_for_function("window.__playground.getSource().includes('DivorceGenerative')")
    assert "DivorceGenerative" in page.evaluate("window.__playground.getSource()")
