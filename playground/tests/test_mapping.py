from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"
COMPILE_TIMEOUT = 30_000
RUN_TIMEOUT = 120_000
SYNTHETIC_LENGTH_SOURCE = """from bayeswire import Data, Observed, model
from bayeswire.distributions import Normal


@model
class SyntheticLength:
    x = Data.vector()
    y = Observed(Normal(x, 1.0))
"""


def fixture_text(relative_path: str) -> str:
    return (FIXTURES / relative_path).read_text()


def open_app(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")


def test_csv_column_mapping(page: Page, base_url: str) -> None:
    open_app(page, base_url)
    page.evaluate(
        "source => window.__playground.setSource(source)",
        fixture_text("divorce/adjusted.py"),
    )
    page.wait_for_function(
        "window.__playground.state().irHash !== ''",
        timeout=COMPILE_TIMEOUT,
    )
    page.locator("#csv-input").set_input_files(FIXTURES / "divorce" / "WaffleDivorce.csv")

    assignments = {
        "M": "Marriage",
        "A": "MedianAgeMarriage",
        "D": "Divorce",
    }
    for input_name in assignments:
        row = page.locator(f'#mapping-table tr[data-input="{input_name}"]')
        expect(row).to_have_attribute("data-status", "missing")
        expect(row.locator(f'select.column-picker[data-input="{input_name}"]')).to_have_count(1)

    for input_name, column_name in assignments.items():
        page.locator(f'select.column-picker[data-input="{input_name}"]').select_option(column_name)

    for input_name in assignments:
        expect(page.locator(f'#mapping-table tr[data-input="{input_name}"]')).to_have_attribute(
            "data-status", "bound"
        )
    expect(page.locator("#run-button")).to_be_enabled()


def test_json_binding_derives_synthetic_length(page: Page, base_url: str) -> None:
    open_app(page, base_url)
    page.evaluate(
        "source => window.__playground.setSource(source)",
        SYNTHETIC_LENGTH_SOURCE,
    )
    page.wait_for_function(
        "window.__playground.state().irHash !== ''",
        timeout=COMPILE_TIMEOUT,
    )
    document = """{
      "format": "bayescycle.data.json.v1",
      "variables": {
        "x": {"dtype": "float64", "shape": [3], "values": [1.0, 2.0, 3.0]},
        "y": {"dtype": "float64", "shape": [3], "values": [1.5, 2.5, 3.5]}
      }
    }"""
    page.locator("#json-input").fill(document)
    page.locator("#json-load").click()

    synthetic = page.locator('#mapping-table tr[data-input="n"]')
    expect(synthetic).to_have_attribute("data-status", "bound")
    expect(synthetic.locator("td").nth(1)).to_have_text("auto:length")
    expect(page.locator("#run-button")).to_be_enabled()


def test_cold_start_example_autoload_and_results_reveal(page: Page, base_url: str) -> None:
    open_app(page, base_url)
    expect(page.locator("#results")).to_be_hidden()
    expect(page.locator("#share-url")).to_be_hidden()

    menu = page.locator("#examples-menu")
    expect(menu.locator("option")).to_have_count(14)
    menu.select_option("eight_schools_non_centered")
    for input_name in ("n_schools", "sigma", "y"):
        expect(page.locator(f'#mapping-table tr[data-input="{input_name}"]')).to_have_attribute(
            "data-status", "bound", timeout=COMPILE_TIMEOUT
        )
    expect(page.locator("#run-button")).to_be_enabled()

    for selector, value in {
        "#chains": "2",
        "#num-warmup": "50",
        "#num-draws": "50",
        "#seed": "17",
    }.items():
        page.locator(selector).fill(value)
    page.locator("#run-button").click()
    page.wait_for_function(
        "window.__playground.state().running === false && "
        "window.__playground.state().lastRun !== null",
        timeout=RUN_TIMEOUT,
    )
    expect(page.locator("#results")).to_be_visible()
