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
    prior_source = "@model\nclass SharedPrior:\n    beta = Param(Normal(3.0, 0.25))\n"
    page.locator("#param-source-other-prior").check()
    page.locator("#prior-source").fill(prior_source)
    page.locator("#share-button").click()
    expect(page.locator("#share-url")).to_be_visible()
    shared_url = page.locator("#share-url").input_value()

    fresh = page.context.browser.new_page()
    try:
        fresh.goto(shared_url)
        expect(fresh.locator("#share-review")).to_be_visible()
        expect(fresh.locator("#share-source")).to_contain_text("LinearRegression")
        expect(fresh.locator("#share-prior-source")).to_contain_text("SharedPrior")
        expect(fresh.locator("#ir-hash")).to_be_hidden()
        fresh.locator("#load-shared").click()
        expect(fresh.locator("#model-source")).to_contain_text("LinearRegression")
        expect(fresh.locator("#inference-seed")).to_have_value("17")
        expect(fresh.locator("#generation-seed")).to_have_value("23")
        expect(fresh.locator("#generation-count")).to_have_value("37")
        expect(fresh.locator("#prior-source")).to_have_value(prior_source)
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


def test_oversized_share_payload_surfaces_on_review_screen(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/#project={'A' * 65537}")
    expect(page.locator("#share-review")).to_be_visible()
    expect(page.locator("#share-source")).to_contain_text(
        "compressed payload exceeds 65536 characters"
    )
    expect(page.locator("#load-shared")).to_be_disabled()


def test_recipient_edits_before_compile_outrank_shared_form_state(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)
    page.locator("#design-json-toggle").click()
    page.locator("#truth-json-toggle").click()
    expect(page.locator("#design-slots")).to_be_visible()
    page.locator("#share-button").click()
    expect(page.locator("#share-url")).to_be_visible()
    shared_url = page.locator("#share-url").input_value()

    fresh = page.context.browser.new_page()
    try:
        fresh.goto(shared_url)
        fresh.locator("#load-shared").click()
        design = '{"x":[10,20]}'
        truth = '{"alpha":9,"beta":8,"sigma":7}'
        fresh.locator("#design-data").fill(design)
        fresh.locator("#truth-data").fill(truth)
        fresh.locator("#compile-button").click()
        expect(fresh.locator("#ir-hash")).to_be_visible(timeout=120_000)
        # The recipient's pre-compile edits stay authoritative over the
        # sender's saved form state.
        expect(fresh.locator("#design-json-field")).to_be_visible()
        expect(fresh.locator("#truth-json-field")).to_be_visible()
        expect(fresh.locator("#design-data")).to_have_value(design)
        expect(fresh.locator("#truth-data")).to_have_value(truth)
    finally:
        fresh.close()

    edited = page.context.browser.new_page()
    try:
        edited.goto(shared_url)
        edited.locator("#load-shared").click()
        source = edited.locator("#model-source").input_value()
        edited.locator("#model-source").fill(source + "\n# recipient tweak\n")
        edited.locator("#compile-button").click()
        expect(edited.locator("#ir-hash")).to_be_visible(timeout=120_000)
        # A pre-compile source edit drops the sender's pending form state.
        expect(edited.locator("#design-json-field")).to_be_visible()
        expect(edited.locator("#truth-json-field")).to_be_visible()
    finally:
        edited.close()


def test_sharing_uncompiled_source_edits_omits_stale_form_state(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)
    page.locator("#design-json-toggle").click()
    page.locator("#truth-json-toggle").click()
    expect(page.locator("#design-slots")).to_be_visible()

    source = page.locator("#model-source").input_value()
    page.locator("#model-source").fill(source + "\n# sender tweak\n")
    page.locator("#share-button").click()
    expect(page.locator("#share-url")).to_be_visible()
    shared_url = page.locator("#share-url").input_value()

    fresh = page.context.browser.new_page()
    try:
        fresh.goto(shared_url)
        fresh.locator("#load-shared").click()
        fresh.locator("#compile-button").click()
        expect(fresh.locator("#ir-hash")).to_be_visible(timeout=120_000)
        # The stale form state was not shared: the recipient sees the
        # carried JSON documents, not forms from the pre-edit schema.
        expect(fresh.locator("#design-json-field")).to_be_visible()
        expect(fresh.locator("#truth-json-field")).to_be_visible()
    finally:
        fresh.close()
