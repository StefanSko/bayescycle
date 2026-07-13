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


MIXED_SOURCE = """from bayeswire import Data, Observed, Param, PartiallyObserved, model
from bayeswire.distributions import Normal


@model
class MixedObservation:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)

    mu = Param(Normal(0.0, 1.0))
    y = PartiallyObserved.vector(
        Normal(mu, 1.0),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
    )
    z = Observed(Normal(mu, 1.0))
"""

MIXED_DOCUMENT = """{
  "format": "bayescycle.data.json.v1",
  "variables": {
    "n": {"dtype": "int64", "shape": [], "values": [5]},
    "n_obs": {"dtype": "int64", "shape": [], "values": [3]},
    "n_mis": {"dtype": "int64", "shape": [], "values": [2]},
    "observed_idx": {"dtype": "int64", "shape": [3], "values": [0, 2, 4]},
    "missing_idx": {"dtype": "int64", "shape": [2], "values": [1, 3]},
    "observed_values": {"dtype": "float64", "shape": [3], "values": [-0.3, 0.8, 1.1]},
    "z": {"dtype": "float64", "shape": [4], "values": [0.1, -0.4, 0.9, 0.3]}
  }
}"""


def test_mixed_partially_observed_gates_predictive_calls(page: Page, base_url: str) -> None:
    """A PartiallyObserved vector plus a regular Observed node must still gate
    the predictive verbs: observed_nodes is non-empty, so the gate has to key
    on the partially-observed flag, not the observed-node count."""
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")
    page.evaluate("source => window.__playground.setSource(source)", MIXED_SOURCE)
    page.wait_for_function(
        "window.__playground.state().irHash !== ''",
        timeout=COMPILE_TIMEOUT,
    )
    page.locator("#json-input").fill(MIXED_DOCUMENT)
    page.locator("#json-load").click()
    expect(page.locator("#run-button")).to_be_enabled()
    for selector, value in {"#chains": "1", "#num-warmup": "10", "#num-draws": "10"}.items():
        page.locator(selector).fill(value)
    page.locator("#run-button").click()
    page.wait_for_function(
        "window.__playground.state().running === false && "
        "window.__playground.state().lastRun !== null",
        timeout=RUN_TIMEOUT,
    )
    expect(page.locator("#run-error")).to_be_hidden()
    for plot_id in ("plot-trank", "plot-esshat", "plot-precis"):
        expect(page.locator(f'#{plot_id} svg[role="img"]')).to_have_count(1)
    expect(page.locator("#plot-ppc")).to_have_text(
        "Posterior predictive display is not available for partially observed models yet."
    )
    expect(page.locator("#plot-overlay")).to_have_text(
        "Prior to posterior display is not available for partially observed models yet."
    )
