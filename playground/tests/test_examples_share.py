from playwright.sync_api import Page, expect


def test_example_and_share_require_explicit_compile(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    expect(page.locator("#examples-menu")).to_be_attached()
    page.locator("#examples-menu").select_option("linear-simulation")
    expect(page.locator("#model-source")).not_to_have_value("")
    expect(page.locator("#observed-data")).to_have_value("")
    expect(page.locator("#ir-hash")).to_be_hidden()
    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)

    page.locator("#inference-seed").fill("17")
    page.locator("#generation-seed").fill("23")
    page.locator("#generation-count").fill("37")
    page.locator("#design-json-toggle").click()
    page.locator("#truth-json-toggle").click()
    expect(page.locator("#design-slots")).to_be_visible()
    expect(page.locator("#parameter-fields")).to_be_visible()
    page.locator("#share-button").click()
    expect(page.locator("#share-url")).to_be_visible()
    shared_url = page.locator("#share-url").input_value()

    fresh = page.context.browser.new_page()
    try:
        fresh.goto(shared_url)
        expect(fresh.locator("#share-review")).to_be_visible()
        expect(fresh.locator("#share-source")).to_contain_text("LinearRegression")
        expect(fresh.locator("#ir-hash")).to_be_hidden()
        fresh.locator("#load-shared").click()
        expect(fresh.locator("#model-source")).to_contain_text("LinearRegression")
        expect(fresh.locator("#inference-seed")).to_have_value("17")
        expect(fresh.locator("#generation-seed")).to_have_value("23")
        expect(fresh.locator("#generation-count")).to_have_value("37")
        expect(fresh.locator("#ir-hash")).to_be_hidden()
        fresh.locator("#compile-button").click()
        expect(fresh.locator("#ir-hash")).to_be_visible(timeout=120_000)
        expect(fresh.locator("#design-slots")).to_be_visible()
        expect(fresh.locator("#parameter-fields")).to_be_visible()
        expect(fresh.locator("#design-expr-x")).to_have_value("[-1,-0.5,0,0.5,1]")
    finally:
        fresh.close()

    old_payload = page.evaluate(
        """async () => {
          const { encodeProject } = await import('/site/src/app/share.mjs');
          return encodeProject({
            v: 1,
            source: document.querySelector('#model-source').value,
            observed: document.querySelector('#observed-data').value,
            design: '',
            truth: '',
            sampler: {
              chains: 1, num_warmup: 0, num_draws: 4, seed: 31,
              target_accept: 0.8, max_treedepth: 10,
            },
          });
        }"""
    )
    old = page.context.browser.new_page()
    try:
        old.goto(f"{base_url}/site/#project={old_payload}")
        old.locator("#load-shared").click()
        expect(old.locator("#inference-seed")).to_have_value("31")
        expect(old.locator("#generation-seed")).to_have_value("31")
        expect(old.locator("#generation-count")).to_have_value("4")
        old.locator("#compile-button").click()
        expect(old.locator("#ir-hash")).to_be_visible(timeout=120_000)
        expect(old.locator("#design-json-field")).to_be_visible()
        expect(old.locator("#truth-json-field")).to_be_visible()
    finally:
        old.close()
