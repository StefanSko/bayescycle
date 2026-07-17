from playwright.sync_api import Page, expect


def test_cancelled_compile_is_actionable_and_allows_a_successful_retry(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    page.locator("#compile-button").click()
    expect(page.locator("#cancel-run")).to_be_visible(timeout=5_000)
    page.locator("#cancel-run").click()

    expect(page.locator("#cancel-run")).to_be_hidden(timeout=5_000)
    expect(page.locator("#compile-error")).to_contain_text("Cancelled", timeout=5_000)
    expect(page.locator("#compile-status")).not_to_contain_text("Compiling")
    expect(page.locator("#compile-button")).to_be_enabled()

    page.locator("#compile-button").click()
    expect(page.locator("#fit-button")).to_be_enabled(timeout=120_000)


def test_cancelled_another_prior_compile_allows_a_successful_retry(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#param-source-other-prior")).to_be_enabled(timeout=120_000)
    page.locator("#param-source-other-prior").check()
    page.locator("#prior-source").fill("while True:\n    pass\n")
    page.locator("#generation-count").fill("1")
    page.locator("#generate-button").click()
    expect(page.locator("#cancel-run")).to_be_visible(timeout=5_000)
    expect(page.locator("#compile-button")).to_be_disabled()
    page.locator("#cancel-run").click()
    expect(page.locator("#run-error")).to_contain_text("Cancelled", timeout=5_000)
    expect(page.locator("#generate-button")).to_be_enabled()

    page.locator("#prior-source").fill(
        "@model\nclass RetryPrior:\n    beta = Param(Normal(3.0, 0.25))\n"
    )
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-generated-datasets")).to_be_visible(timeout=120_000)


def test_cancelled_fit_returns_quickly_and_allows_a_successful_retry(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    page.locator("#compile-button").click()
    expect(page.locator("#fit-button")).to_be_enabled(timeout=120_000)

    page.locator("#chains").fill("1")
    page.locator("#warmup").fill("100000")
    page.locator("#draws").fill("100000")
    page.locator("#max-treedepth").fill("20")
    page.locator("#fit-button").click()
    expect(page.locator("#cancel-run")).to_be_visible(timeout=5_000)
    page.locator("#cancel-run").click()

    expect(page.locator("#cancel-run")).to_be_hidden(timeout=5_000)
    expect(page.locator("#run-error")).to_contain_text("Cancelled", timeout=5_000)
    expect(page.locator("#fit-button")).to_be_enabled(timeout=5_000)

    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("4")
    page.locator("#max-treedepth").fill("4")
    page.locator("#fit-button").click()
    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
