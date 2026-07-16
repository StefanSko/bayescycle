from playwright.sync_api import Page, expect


def test_successful_compile_is_announced_without_relying_on_hash(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    page.locator("#compile-button").click()
    expect(page.locator("#compile-status")).to_have_text(
        "Compiled. 3 parameters, 2 design slots, 1 observed slot.", timeout=120_000
    )
    expect(page.locator("#ir-hash")).to_be_visible()


def test_authoring_edits_made_during_recompile_survive_completion(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)
    page.locator("#design-json-toggle").click()
    page.locator("#truth-json-toggle").click()

    page.locator("#compile-button").click()
    expect(page.locator("#compile-status")).to_have_text("Compiling in an isolated worker…")
    design = '{"x":[10,20]}'
    truth = '{"alpha":1,"beta":2,"sigma":3}'
    page.locator("#design-data").fill(design)
    page.locator("#truth-data").fill(truth)

    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)
    expect(page.locator("#design-json-field")).to_be_visible()
    expect(page.locator("#truth-json-field")).to_be_visible()
    expect(page.locator("#design-data")).to_have_value(design)
    expect(page.locator("#truth-data")).to_have_value(truth)
