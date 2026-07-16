from playwright.sync_api import Page, expect


def test_oversized_example_sidecar_is_rejected_before_install(page: Page, base_url: str) -> None:
    page.route(
        "**/linear_regression.design.json",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body="x" * (4 * 1024 * 1024 + 1),
        ),
    )
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    expect(page.locator("#compile-error")).to_contain_text(
        "linear_regression.design.json exceeds 4194304 UTF-8 bytes"
    )
    expect(page.locator("#model-source")).to_have_value("")
    expect(page.locator("#design-data")).to_have_value("")


def test_stale_example_load_cannot_overwrite_latest_selection(page: Page, base_url: str) -> None:
    page.add_init_script(
        """
        const nativeFetch = globalThis.fetch;
        globalThis.fetch = async (...args) => {
          const url = String(args[0]);
          if (url.includes("eight_schools_non_centered")) {
            await new Promise((resolve) => setTimeout(resolve, 300));
          }
          return nativeFetch(...args);
        };
        """
    )
    page.goto(f"{base_url}/site/")
    menu = page.locator("#examples-menu")
    expect(menu.locator("option")).to_have_count(4)
    menu.select_option("eight-schools")
    menu.select_option("linear-simulation")
    expect(page.locator("#model-source")).to_contain_text("LinearRegression")
    page.wait_for_timeout(500)
    expect(menu).to_have_value("linear-simulation")
    expect(page.locator("#model-source")).to_contain_text("LinearRegression")
    expect(page.locator("#model-source")).not_to_contain_text("EightSchools")
