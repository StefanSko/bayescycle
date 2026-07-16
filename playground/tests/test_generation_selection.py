from playwright.sync_api import Page, expect


def test_generated_pair_selection_conditioning_and_recovery(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#chains").fill("1")
    page.locator("#warmup").fill("8")
    page.locator("#draws").fill("8")
    page.locator("#generation-count").fill("2")
    page.locator("#compile-button").click()

    expect(page.locator("#generate-button")).to_be_enabled(timeout=120_000)
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-generated-datasets")).to_be_visible(timeout=120_000)
    expect(page.locator("#generated-dataset-index")).to_be_enabled()
    expect(page.locator("#generated-dataset-index option")).to_have_count(2)
    expect(page.locator("#selected-pair-summary")).to_contain_text("Pair 0 of 2")
    expect(page.locator("#dataset-source-generated")).to_be_enabled()

    page.locator("#generated-dataset-index").select_option("1")
    expect(page.locator("#selected-pair-summary")).to_contain_text("Pair 1 of 2")
    page.locator("#dataset-source-generated").check()
    expect(page.locator("#fit-button")).to_have_text("Fit simulated pair 1")
    page.locator("#fit-button").click()
    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-recovery")).to_be_visible(timeout=120_000)
    expect(page.locator("#recovery-summary")).to_be_visible()

    page.locator("#generated-dataset-index").select_option("0")
    expect(page.locator("#artifact-generated-datasets")).to_be_visible()
    expect(page.locator("#artifact-posterior")).to_be_hidden()
    expect(page.locator("#artifact-recovery")).to_be_hidden()
    expect(page.locator("#dataset-source-generated")).to_be_enabled()
