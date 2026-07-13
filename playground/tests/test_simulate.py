from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"
COMPILE_TIMEOUT = 30_000
PRIOR_TIMEOUT = 60_000
SIMULATE_TIMEOUT = 60_000
SAMPLE_TIMEOUT = 180_000
PARAMETERS = ("alpha", "beta_m", "beta_a", "sigma")


def open_generative_model(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")
    source = (FIXTURES / "divorce" / "generative.py").read_text()
    page.evaluate("source => window.__playground.setSource(source)", source)
    page.wait_for_function("window.__playground.state().irHash !== ''", timeout=COMPILE_TIMEOUT)


def select_design_mode(page: Page) -> None:
    observed = page.locator("#data-mode-observed")
    design = page.locator("#data-mode-design")
    expect(observed).to_be_checked()
    design.check()
    expect(design).to_be_checked()
    expect(page.locator("#design-panel")).to_be_visible()


def assert_design_defaults(page: Page) -> None:
    rows = page.locator("#design-values tr[data-design-name]")
    expect(rows).to_have_count(2)
    assert set(rows.evaluate_all("rows => rows.map(row => row.dataset.designName)")) == {
        "M",
        "A",
    }
    for name in ("M", "A"):
        row = page.locator(f'#design-values tr[data-design-name="{name}"]')
        expect(row.locator('[data-design-field="low"]')).to_have_value("0.5")
        expect(row.locator('[data-design-field="high"]')).to_have_value("1.5")
        expect(row.locator('[data-design-field="n"]')).to_have_value("50")

    truth_rows = page.locator("#truth-values tr[data-truth-name]")
    expect(truth_rows).to_have_count(4)
    defaults = {"alpha": "0", "beta_m": "0", "beta_a": "0", "sigma": "1"}
    for name, value in defaults.items():
        expect(
            page.locator(f'#truth-values tr[data-truth-name="{name}"] input[type="number"]')
        ).to_have_value(value)


def test_prior_predictive_with_zero_data(page: Page, base_url: str) -> None:
    open_generative_model(page, base_url)
    select_design_mode(page)
    assert_design_defaults(page)

    state = page.evaluate("window.__playground.state()")
    assert state["mappingComplete"] is False
    expect(page.locator("#run-prior-predictive")).to_be_enabled()
    page.locator("#run-prior-predictive").click()
    expect(page.locator("#plot-ppc svg")).to_have_count(1, timeout=PRIOR_TIMEOUT)
    assert page.evaluate("window.__playground.state().dataMode") == "design"


def test_simulate_then_recover(page: Page, base_url: str) -> None:
    open_generative_model(page, base_url)
    select_design_mode(page)
    assert_design_defaults(page)

    # This test's subject is slope identifiability under decorrelated designs,
    # not the default ranges — set a centered design explicitly (centered
    # covariates keep alpha decoupled from the slopes), and pick truths the
    # model's own priors (alpha ~ N(0, 0.2), betas ~ N(0, 0.5)) can reach
    # without fighting shrinkage.
    for name in ("M", "A"):
        row = page.locator(f'#design-values tr[data-design-name="{name}"]')
        row.locator('[data-design-field="low"]').fill("-1")
        row.locator('[data-design-field="high"]').fill("1")

    truth = {"alpha": 0.2, "beta_m": -0.5, "beta_a": 0.5, "sigma": 1.0}
    for name, value in truth.items():
        page.locator(f'#truth-values tr[data-truth-name="{name}"] input[type="number"]').fill(
            str(value)
        )
    for selector, value in {
        "#chains": "2",
        "#num-warmup": "200",
        "#num-draws": "200",
        "#seed": "33",
    }.items():
        page.locator(selector).fill(value)

    expect(page.locator("#run-simulate")).to_be_enabled()
    page.locator("#run-simulate").click()
    for name in ("M", "A", "n", "D"):
        row = page.locator(f'#mapping-table tr[data-input="{name}"]')
        expect(row).to_have_attribute("data-status", "bound", timeout=SIMULATE_TIMEOUT)
        expect(row.locator("td").nth(1)).to_have_text("simulated")

    run_button = page.locator("#run-button")
    expect(run_button).to_be_enabled()
    expect(run_button).to_have_text("Sample on simulated data")
    assert page.evaluate("window.__playground.state().simulated") is True
    run_button.click()
    page.wait_for_function(
        "window.__playground.state().running === false && "
        "window.__playground.state().recovery !== null",
        timeout=SAMPLE_TIMEOUT,
    )

    recovery = page.evaluate("window.__playground.state().recovery")
    assert set(recovery) == set(PARAMETERS)
    for name, value in truth.items():
        entry = recovery[name]
        assert entry["truth"] == value
        assert entry["low"] <= entry["high"]
        assert entry["inside"] == (entry["low"] <= value <= entry["high"])
    # An 89% interval misses truth ~11% of the time per parameter even for a
    # perfectly calibrated sampler, so all-four-inside is not guaranteed. The
    # decorrelated design must make both slopes identifiable (the point of
    # this test), and coverage must hold for the clear majority.
    assert recovery["beta_m"]["inside"] is True
    assert recovery["beta_a"]["inside"] is True
    assert sum(1 for entry in recovery.values() if entry["inside"]) >= 3

    markers = page.locator("#plot-precis svg line.truth-marker")
    expect(markers).to_have_count(4)
    assert set(markers.evaluate_all("lines => lines.map(line => line.dataset.parameter)")) == set(
        PARAMETERS
    )
