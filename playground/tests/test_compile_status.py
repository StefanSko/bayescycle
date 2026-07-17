from playwright.sync_api import Page, expect


def test_successful_compile_is_announced_without_relying_on_hash(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    page.locator("#compile-button").click()
    expect(page.locator("#compile-status")).to_have_text(
        "Compiled. 3 parameters, 2 design slots, 1 observed slot.", timeout=120_000
    )
    expect(page.locator("#ir-hash")).to_be_visible()


def test_oversized_model_source_surfaces_without_starting_a_compile(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill("é" * (1024 * 1024 // 2 + 1))
    page.locator("#compile-button").click()
    expect(page.locator("#compile-error")).to_contain_text("maximum UTF-8 size of 1048576 bytes")
    expect(page.locator("#compile-button")).to_be_enabled()
    page.locator("#share-button").click()
    expect(page.locator("#share-error")).to_contain_text(
        "decompressed payload exceeds 1048576 bytes"
    )
    expect(page.locator("#share-output")).to_be_hidden()


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
