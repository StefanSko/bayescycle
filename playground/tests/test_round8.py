from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"
COMPILE_TIMEOUT = 30_000
SIMULATE_TIMEOUT = 60_000
RUN_TIMEOUT = 180_000
PARTIAL_REASON = (
    "Design mode does not support partially observed models yet — bind observed data instead."
)


def open_model(page: Page, base_url: str, fixture_name: str) -> None:
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")
    source = (FIXTURES / "corpus" / fixture_name).read_text()
    page.evaluate("source => window.__playground.setSource(source)", source)
    page.wait_for_function(
        "window.__playground.state().irHash !== ''",
        timeout=COMPILE_TIMEOUT,
    )


def test_ordered_recovery_uses_coordinate_truths(page: Page, base_url: str) -> None:
    open_model(page, base_url, "ordinal_regression.py")
    page.locator("#data-mode-design").check()
    page.locator("#run-simulate").click()
    page.wait_for_function(
        "window.__playground.state().simulated === true",
        timeout=SIMULATE_TIMEOUT,
    )
    for selector, value in {
        "#chains": "1",
        "#num-warmup": "50",
        "#num-draws": "50",
    }.items():
        page.locator(selector).fill(value)
    page.locator("#run-button").click()
    page.wait_for_function(
        "window.__playground.state().running === false && "
        "window.__playground.state().recovery !== null",
        timeout=RUN_TIMEOUT,
    )

    recovery = page.evaluate("window.__playground.state().recovery")
    cutpoints = sorted(
        (
            (int(name.removeprefix("cutpoints[").removesuffix("]")), entry)
            for name, entry in recovery.items()
            if name.startswith("cutpoints[")
        ),
        key=lambda item: item[0],
    )
    assert len(cutpoints) == 2
    truths = [entry["truth"] for _, entry in cutpoints]
    assert truths[0] < truths[1]
    for _, entry in cutpoints:
        assert entry["inside"] == (entry["low"] <= entry["truth"] <= entry["high"])
    expect(page.locator("#plot-precis svg line.truth-marker")).to_have_count(len(recovery))
    expect(page.locator("#run-error")).to_be_hidden()


def test_partially_observed_gates_and_observed_run(page: Page, base_url: str) -> None:
    open_model(page, base_url, "partially_observed_mvn.py")
    page.locator("#data-mode-design").check()
    expect(page.locator("#run-prior-predictive")).to_be_disabled()
    expect(page.locator("#run-simulate")).to_be_disabled()
    expect(page.locator("#simulate-reason")).to_have_text(PARTIAL_REASON)
    expect(page.locator("#simulate-reason")).to_be_visible()

    page.locator("#data-mode-observed").check()
    data = (FIXTURES / "corpus" / "partially_observed_mvn.data.json").read_text()
    page.locator("#json-input").fill(data)
    page.locator("#json-load").click()
    expect(page.locator("#run-button")).to_be_enabled()
    for selector, value in {
        "#chains": "1",
        "#num-warmup": "50",
        "#num-draws": "50",
    }.items():
        page.locator(selector).fill(value)
    page.locator("#run-button").click()
    page.wait_for_function(
        "window.__playground.state().running === false && "
        "window.__playground.state().lastRun !== null",
        timeout=RUN_TIMEOUT,
    )

    for plot_id in ("plot-trank", "plot-esshat", "plot-precis"):
        expect(page.locator(f'#{plot_id} svg[role="img"]')).to_have_count(1)
    # Prior draws are unavailable for partially observed models, so both
    # predictive panels show an honest note instead of a fabricated curve.
    expect(page.locator("#plot-ppc")).to_have_text(
        "Posterior predictive display is not available for partially observed models yet."
    )
    expect(page.locator("#plot-overlay")).to_have_text(
        "Prior to posterior display is not available for partially observed models yet."
    )
    expect(page.locator("#run-error")).to_be_hidden()
