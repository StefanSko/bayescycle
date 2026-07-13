from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"


def test_explicit_documents_simulate_sample_and_check_recovery(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    expect(page.locator("#simulate-button")).to_be_attached()
    source = """from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Exponential, Normal

@model
class Generative:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Exponential(1.0), constraint=Positive())
    x = Data.vector()
    y = Observed(Normal(alpha + beta * x, sigma))
"""
    page.locator("#model-source").fill(source)
    page.locator("#design-data").fill('{"x": [-1, -0.5, 0, 0.5, 1]}')
    page.locator("#truth-data").fill('{"alpha": 0.2, "beta": 0.6, "sigma": 0.8}')
    page.locator("#compile-button").click()
    expect(page.locator("#simulate-button")).to_be_enabled(timeout=120_000)
    page.locator("#chains").fill("2")
    page.locator("#warmup").fill("8")
    page.locator("#draws").fill("12")
    page.locator("#simulate-button").click()
    expect(page.locator("#artifact-simulated")).to_be_visible(timeout=120_000)
    expect(page.locator("#sample-simulated-button")).to_be_enabled()
    page.locator("#sample-simulated-button").click()
    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-recovery")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-simulated")).to_be_visible()
