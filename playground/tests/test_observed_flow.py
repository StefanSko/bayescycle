from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"


def test_observed_model_to_artifacts(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    source = (FIXTURES / "corpus" / "eight_schools_non_centered.py").read_text()
    data = (FIXTURES / "engine" / "eight_schools_non_centered" / "data.json").read_text()

    page.locator("#model-source").fill(source)
    page.locator("#observed-data").fill(data)
    expect(page.locator("#design-data")).to_have_value("")
    expect(page.locator("#ir-hash")).to_be_hidden()

    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_have_text(
        "sha256:6cb101cf5159bddcbe10650a87a8763054a462da0f4374b8b61a2a1a861695dc",
        timeout=120_000,
    )

    page.locator("#chains").fill("2")
    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("4")
    expect(page.locator("#fit-button")).to_be_enabled()
    page.locator("#fit-button").click()
    expect(page.locator("#run-status")).to_have_text("Fitting is running…")
    expect(page.locator("#compile-button")).to_be_disabled()

    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-diagnostics")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-model")).to_be_visible()
    expect(page.locator("#artifact-data")).to_be_visible()
    expect(page.locator("#progress")).to_contain_text("chain 1")
    expect(page.locator("#run-error")).to_be_hidden()

    expect(page.locator("#param-source-posterior")).to_be_enabled()
    page.locator("#param-source-posterior").check()
    expect(page.locator("#generate-button")).to_have_text("Generate from posterior")
    expect(page.locator("#generate-button")).to_be_enabled()
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-posterior-predictive")).to_be_visible(timeout=120_000)

    page.locator("#generation-seed").fill("1")
    expect(page.locator("#artifact-posterior-predictive")).to_be_hidden()
    expect(page.locator("#artifact-posterior")).to_be_visible()
    expect(page.locator("#param-source-posterior")).to_be_enabled()
    expect(page.locator("#param-source-posterior")).to_be_checked()

    page.locator("#examples-menu").select_option("linear-simulation")
    expect(page.locator("#progress")).to_be_empty()
