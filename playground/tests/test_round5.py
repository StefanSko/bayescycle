from playwright.sync_api import Page, expect
from test_shell import PLOT_IDS, fixture_text, prepare_run, set_source, start_and_wait_for_run
from test_simulate import open_generative_model, select_design_mode

COMPILE_TIMEOUT = 30_000
PRIOR_TIMEOUT = 60_000
SIMULATE_TIMEOUT = 60_000


def test_source_edit_clears_completed_results(page: Page, base_url: str) -> None:
    prepare_run(page, base_url)
    start_and_wait_for_run(page)
    expect(page.locator("#results")).to_be_visible()

    set_source(page, fixture_text("divorce/naive.py"))
    expect(page.locator("#results")).to_be_hidden()
    for plot_id in PLOT_IDS:
        expect(page.locator(f"#{plot_id}")).to_be_empty()
    expect(page.locator("#download-fit")).to_be_disabled()
    expect(page.locator("#download-diagnostics")).to_be_disabled()


def test_truth_edit_invalidates_simulated_data(page: Page, base_url: str) -> None:
    open_generative_model(page, base_url)
    select_design_mode(page)
    page.locator("#run-simulate").click()
    page.wait_for_function(
        "window.__playground.state().simulated === true",
        timeout=SIMULATE_TIMEOUT,
    )
    expect(page.locator("#run-button")).to_have_text("Sample on simulated data")
    expect(page.locator("#run-button")).to_be_enabled()

    page.locator('#truth-values tr[data-truth-name="alpha"] input[type="number"]').fill("0.75")
    assert page.evaluate("window.__playground.state().simulated") is False
    expect(page.locator("#run-button")).to_have_text("Run sampler")
    expect(page.locator("#run-button")).to_be_disabled()


def test_scalar_design_default_supports_prior_predictive(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")
    set_source(page, fixture_text("corpus/varying_intercepts_poisson.py"))
    page.wait_for_function(
        "window.__playground.state().irHash !== ''",
        timeout=COMPILE_TIMEOUT,
    )
    page.locator("#data-mode-design").check()

    scalar = page.locator(
        '#design-values tr[data-design-name="n_groups"] [data-design-field="value"]'
    )
    expect(scalar).to_have_value("2")
    expect(scalar).to_be_editable()
    expect(page.locator("#run-prior-predictive")).to_be_enabled()
    page.locator("#run-prior-predictive").click()
    expect(page.locator('#plot-ppc svg[role="img"]')).to_have_count(
        1,
        timeout=PRIOR_TIMEOUT,
    )
    expect(page.locator("#run-error")).to_be_hidden()
