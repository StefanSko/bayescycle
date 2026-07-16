from playwright.sync_api import Page, expect


def test_examples_load_raw_documents_in_json_escape_hatches(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    menu = page.locator("#examples-menu")
    expect(menu.locator("option")).to_have_count(4)
    menu.select_option("eight-schools")
    expect(page.locator("#design-data")).to_have_value("")
    menu.select_option("linear-simulation")
    expect(page.locator("#design-data")).not_to_have_value("")
    expect(page.locator("#truth-data")).not_to_have_value("")
    page.locator("#compile-button").click()
    expect(page.locator("#design-json-field")).to_be_visible(timeout=120_000)
    expect(page.locator("#truth-json-field")).to_be_visible()
    expect(page.locator("#design-slots")).to_be_hidden()


def test_scalar_design_slots_remain_in_the_json_escape_hatch(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("ordinal-simulation")
    expect(page.locator("#design-data")).not_to_have_value("")
    original = page.locator("#design-data").input_value()
    page.locator("#compile-button").click()
    expect(page.locator("#design-json-field")).to_be_visible(timeout=120_000)
    expect(page.locator("#design-json-toggle")).to_be_disabled()
    expect(page.locator("#design-data")).to_have_value(original)


def test_entering_the_parameter_form_rewrites_a_stale_truth_document(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#truth-json-field")).to_be_visible(timeout=120_000)
    page.locator("#truth-data").fill('{"alpha": 1}')
    expect(page.locator("#generate-button")).to_be_enabled()

    page.locator("#truth-json-toggle").click()
    expect(page.locator("#fixed-value-alpha")).to_have_value("1")
    expect(page.locator("#fixed-value-sigma")).to_have_value("")
    # The hidden document now follows the form: sigma is missing, so the
    # stale {"alpha": 1} JSON must not remain simulatable.
    expect(page.locator("#generate-button")).to_be_disabled()

    page.locator("#fixed-value-beta").fill("0.6")
    page.locator("#fixed-value-sigma").fill("0.8")
    expect(page.locator("#generate-button")).to_be_enabled()
