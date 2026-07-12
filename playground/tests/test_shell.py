from __future__ import annotations

import json
import re
from pathlib import Path

from conftest import OriginIsolation
from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"
COMPILE_TIMEOUT = 30_000
RUN_TIMEOUT = 120_000
EIGHT_SCHOOLS_HASH = "6cb101cf5159bddcbe10650a87a8763054a462da0f4374b8b61a2a1a861695dc"
BROKEN_SOURCE = """from bayeswire import Param, model
from bayeswire.distributions import Normal

class Base:
    pass

@model
class Invalid(Base):
    x = Param(Normal(0.0, 1.0))
"""
BROKEN_MESSAGE = (
    "Model declaration classes must not use inheritance: 'Invalid' inherits from 'Base'. "
    "All declarations must live in the decorated class body; inherited declarations would be "
    "silently ignored otherwise"
)
PLOT_IDS = ["plot-trank", "plot-esshat", "plot-precis", "plot-ppc", "plot-overlay"]


def fixture_text(relative_path: str) -> str:
    return (FIXTURES / relative_path).read_text()


def open_app(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/index.html")
    page.wait_for_function("window.__playground !== undefined")


def set_source(page: Page, source: str) -> None:
    page.evaluate("source => window.__playground.setSource(source)", source)


def compile_eight_schools(page: Page) -> None:
    hashes = json.loads(fixture_text("corpus/hashes.json"))
    assert hashes["eight_schools_non_centered"] == EIGHT_SCHOOLS_HASH
    set_source(page, fixture_text("corpus/eight_schools_non_centered.py"))
    expect(page.locator("#ir-hash-chip")).to_have_text(
        EIGHT_SCHOOLS_HASH,
        timeout=COMPILE_TIMEOUT,
    )


def bind_eight_schools(page: Page) -> None:
    page.locator("#json-input").fill(fixture_text("engine/eight_schools_non_centered/data.json"))
    page.locator("#json-load").click()
    for name in ("n_schools", "sigma", "y"):
        expect(page.locator(f'#mapping-table tr[data-input="{name}"]')).to_have_attribute(
            "data-status",
            "bound",
        )


def configure_small_run(page: Page) -> None:
    for selector, value in {
        "#chains": "2",
        "#num-warmup": "50",
        "#num-draws": "50",
        "#seed": "17",
    }.items():
        page.locator(selector).fill(value)


def prepare_run(page: Page, base_url: str) -> None:
    open_app(page, base_url)
    compile_eight_schools(page)
    bind_eight_schools(page)
    expect(page.locator("#run-button")).to_be_enabled()
    configure_small_run(page)


def start_and_wait_for_run(page: Page, prove_streaming: bool = False) -> None:
    page.locator("#run-button").click()
    if prove_streaming:
        page.wait_for_function(
            """() => window.__playground.state().running &&
                [...document.querySelectorAll('.chain-progress')].some(element => {
                  const match = element.textContent.match(/(\\d+) draws/);
                  return match && Number(match[1]) > 0;
                })""",
            timeout=RUN_TIMEOUT,
        )
    page.wait_for_function(
        "window.__playground.state().running === false && "
        "window.__playground.state().lastRun !== null",
        timeout=RUN_TIMEOUT,
    )


def downloaded_bytes(page: Page, selector: str) -> bytes:
    with page.expect_download(timeout=COMPILE_TIMEOUT) as download_info:
        page.locator(selector).click()
    return Path(download_info.value.path()).read_bytes()


def test_layout_and_a11y(page: Page, base_url: str) -> None:
    open_app(page, base_url)

    main = page.locator("main")
    expect(main).to_have_count(1)
    expect(main.locator(":scope > section")).to_have_count(3)
    for pane_id in ("pane-model", "pane-data", "pane-run"):
        pane = page.locator(f"#{pane_id}")
        expect(pane).to_have_count(1)
        assert pane.evaluate(
            "element => Boolean(element.getAttribute('aria-labelledby') || "
            "element.querySelector('h2'))"
        )
    assert main.evaluate("element => getComputedStyle(element).display") == "grid"
    assert page.locator("html").get_attribute("lang")
    expect(page.locator("h1")).to_have_count(1)
    expect(page.locator("section h2")).to_have_count(3)

    expect(page.locator('[data-testid="model-editor"]')).to_have_count(1)
    hash_chip = page.locator("#ir-hash-chip")
    expect(hash_chip).to_have_count(1)
    assert not hash_chip.is_visible() or hash_chip.text_content() == ""
    expect(page.locator("#compile-error")).to_be_hidden()

    expect(page.locator('#csv-input[type="file"]')).to_have_count(1)
    expect(page.locator("textarea#json-input")).to_have_count(1)
    expect(page.locator("#json-load")).to_have_count(1)
    expect(page.locator("table#mapping-table")).to_have_count(1)

    expect(page.locator("form#sampler-settings")).to_have_count(1)
    expect(page.locator("#run-button")).to_be_disabled()
    expect(page.locator("div#progress")).to_have_count(1)
    expect(page.locator("div#results")).to_have_count(1)
    for plot_id in PLOT_IDS:
        expect(page.locator(f"div#{plot_id}")).to_have_count(1)
        plot_name = plot_id.removeprefix("plot-")
        expect(page.locator(f'.download-svg[data-plot="{plot_name}"]')).to_have_count(1)
    expect(page.locator("#plot-mode option")).to_have_text(["rank", "trace"])
    expect(page.locator("#download-fit")).to_have_count(1)
    expect(page.locator("#download-diagnostics")).to_have_count(1)

    defaults = {
        "chains": "4",
        "num-warmup": "1000",
        "num-draws": "2000",
        "seed": "0",
        "target-accept": "0.8",
        "max-treedepth": "10",
    }
    for input_id, value in defaults.items():
        field = page.locator(f"#{input_id}")
        expect(field).to_have_value(value)
        assert field.evaluate("element => element.labels !== null && element.labels.length > 0"), (
            f"#{input_id} has no associated label"
        )

    unlabeled_inputs = page.locator("form input").evaluate_all(
        "elements => elements.filter(element => !element.labels || !element.labels.length)"
        ".map(element => element.id)"
    )
    assert not unlabeled_inputs, f"form inputs without labels: {unlabeled_inputs}"
    unnamed_buttons = page.locator("button").evaluate_all(
        "elements => elements.filter(element => "
        "!(element.innerText.trim() || element.getAttribute('aria-label')))"
        ".map(element => element.id)"
    )
    assert not unnamed_buttons, f"buttons without accessible names: {unnamed_buttons}"


def test_author_compile_flow(page: Page, base_url: str) -> None:
    open_app(page, base_url)
    source = fixture_text("corpus/eight_schools_non_centered.py")
    compile_eight_schools(page)
    assert page.evaluate("window.__playground.getSource()") == source
    expect(page.locator("#compile-error")).to_be_hidden()

    set_source(page, BROKEN_SOURCE)
    assert page.evaluate("window.__playground.getSource()") == BROKEN_SOURCE
    error = page.locator("#compile-error")
    expect(error).to_contain_text(BROKEN_MESSAGE, timeout=COMPILE_TIMEOUT)
    expect(error).to_contain_text("Traceback (most recent call last):")
    expect(error).to_contain_text("bayeswire/model/decorator.py")
    expect(error).to_be_visible()
    expect(page.locator("#run-button")).to_be_disabled()


def test_bind_flow(page: Page, base_url: str) -> None:
    open_app(page, base_url)
    compile_eight_schools(page)
    bind_eight_schools(page)
    for name in ("sigma", "y"):
        expect(page.locator(f'input.standardize-toggle[data-column="{name}"]')).to_have_count(1)
    expect(page.locator("#run-button")).to_be_enabled()


def test_run_flow(page: Page, base_url: str) -> None:
    prepare_run(page, base_url)
    start_and_wait_for_run(page, prove_streaming=True)

    expect(page.locator(".chain-progress")).to_have_count(2)
    for index in range(2):
        expect(page.locator(f'.chain-progress[data-chain="{index}"]')).to_contain_text(
            re.compile(r"\d+ draws · \d+ divergent")
        )
    for plot_id in PLOT_IDS:
        svg = page.locator(f"#{plot_id} svg[role='img']")
        expect(svg).to_have_count(1)
        assert svg.get_attribute("aria-label"), f"#{plot_id} SVG has no aria-label"

    rank_svg = page.locator("#plot-trank svg").evaluate("element => element.outerHTML")
    page.locator("#plot-mode").select_option("trace")
    page.wait_for_function(
        "oldSvg => document.querySelector('#plot-trank svg').outerHTML !== oldSvg",
        arg=rank_svg,
    )
    trace_svg = page.locator("#plot-trank svg").evaluate("element => element.outerHTML")
    assert trace_svg != rank_svg

    state = page.evaluate("window.__playground.state()")
    assert state["lastRun"]["seed"] == 17
    assert state["lastRun"]["chains"] == 2
    assert state["lastRun"]["drawsStreamed"] > 0
    assert isinstance(state["lastRun"]["divergences"], int)
    assert state["lastRun"]["divergences"] >= 0


def test_downloads_are_byte_stable(page: Page, base_url: str) -> None:
    prepare_run(page, base_url)
    start_and_wait_for_run(page)
    first_fit = downloaded_bytes(page, "#download-fit")
    first_diagnostics = downloaded_bytes(page, "#download-diagnostics")
    # The fit download must be one valid NDJSON stream (the merged chains):
    # no blank lines, every line a JSON object, both chains present.
    fit_lines = first_fit.decode().splitlines()
    assert fit_lines, "fit download is empty"
    assert all(line.strip() for line in fit_lines), "fit download contains blank lines"
    header = json.loads(fit_lines[0])
    assert header["chain_count"] == 2
    for line in fit_lines[1:]:
        json.loads(line)
    json.loads(first_diagnostics)

    start_and_wait_for_run(page)
    second_fit = downloaded_bytes(page, "#download-fit")
    second_diagnostics = downloaded_bytes(page, "#download-diagnostics")
    assert second_fit == first_fit
    assert second_diagnostics == first_diagnostics

    svg_bytes = downloaded_bytes(page, '.download-svg[data-plot="trank"]')
    assert svg_bytes.startswith(b"<svg")


def test_zero_third_party(
    page: Page,
    base_url: str,
    origin_isolation: OriginIsolation,
) -> None:
    prepare_run(page, base_url)
    start_and_wait_for_run(page, prove_streaming=True)
    origin_isolation.assert_only_origin()
