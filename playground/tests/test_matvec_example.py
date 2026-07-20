from playwright.sync_api import Page, expect


def test_correlated_mvn_example_compiles_and_fits_with_wasm_engine(
    page: Page,
    base_url: str,
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("mvn-non-centered")

    expect(page.locator("#model-source")).to_contain_text("linear(latent_chol)")
    expect(page.locator("#observed-data")).to_contain_text('"latent_chol"')
    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_have_text(
        "sha256:e7f77bcec5f3bdc273499262b2d09151bf9e52dcf293d74ffe6896cc0aa2a6a9",
        timeout=120_000,
    )

    page.locator("#chains").fill("1")
    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("4")
    expect(page.locator("#fit-button")).to_be_enabled()
    page.locator("#fit-button").click()

    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-diagnostics")).to_be_visible(timeout=120_000)
    expect(page.locator("#run-error")).to_be_hidden()
