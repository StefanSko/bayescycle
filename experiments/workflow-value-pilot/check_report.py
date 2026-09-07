"""Smoke-test the standalone report offline, on desktop and a narrow viewport."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent


def main() -> None:
    errors: list[str] = []
    remote_requests: list[str] = []
    expected = json.loads((ROOT / "evaluator/evaluation.json").read_text())
    with tempfile.TemporaryDirectory() as temporary, sync_playwright() as playwright:
        # The copied page has no neighboring evidence files: all views must work
        # from the embedded snapshot, even with networking disabled.
        report = Path(temporary) / "report.html"
        shutil.copyfile(ROOT / "report.html", report)
        browser = playwright.chromium.launch()
        context = browser.new_context(offline=True, viewport={"width": 1440, "height": 1050})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                remote_requests.append(request.url)
                if request.url.startswith(("http:", "https:"))
                else None
            ),
        )
        page.goto(report.as_uri())
        assert page.locator("#fitCount").inner_text() == "12"
        assert page.locator("#sessionCount").inner_text() == "9"
        assert page.locator("#handoffCount").inner_text() == "3 / 3"
        assert page.locator("#fitTable tr").count() == 3
        assert "0.5880" in page.locator("#fitTable").inner_text()
        page.locator("#journey-tab-0").focus()
        page.keyboard.press("ArrowRight")
        assert page.locator("#journey-tab-1").get_attribute("aria-selected") == "true"
        assert "Three independent" in page.locator("#journeyPanel").inner_text()
        page.locator("#journey-tab-0").click()
        page.locator("#priorRevised").click()
        assert "0.5613" in page.locator("#fitTable").inner_text()
        page.locator("#dataset").select_option("recovery")
        assert page.locator("#showTruth").is_checked()
        assert "truth 0.4" in page.locator("#forest").text_content()
        assert "misses the known" in page.locator("#resultNote").inner_text()
        assert "0.2635" in page.locator("#fitTable").inner_text()
        page.locator("#priorInitial").click()
        assert "0.2757" in page.locator("#fitTable").inner_text()
        page.locator("#dataset").select_option("primary")
        assert not page.locator("#showTruth").is_checked()
        page.locator("#showTruth").check()
        assert "truth 0.65" in page.locator("#forest").text_content()
        page.locator("#effortMetric").select_option("calls")
        assert page.locator(".effort-value").all_text_contents() == ["56", "73", "91"]
        for stage in ("revision", "handoff"):
            page.locator(f'#effortStages input[value="{stage}"]').uncheck()
        assert page.locator(".effort-value").all_text_contents() == ["19", "35", "51"]
        page.locator('#effortStages input[value="initial"]').uncheck()
        assert "No sessions selected" in page.locator("#effortCaption").inner_text()
        for stage in ("initial", "revision", "handoff"):
            page.locator(f'#effortStages input[value="{stage}"]').check()
        page.locator("#effortMetric").select_option("time")
        for arm in ("a-numpyro", "b-bayesjax", "c-bayescycle"):
            page.locator("#figureArm").select_option(arm)
            for stage in ("initial", "revised"):
                page.locator("#figureStage").select_option(stage)
                for kind in ("prior", "trace", "predictive"):
                    page.locator("#figureKind").select_option(kind)
                    page.wait_for_function(
                        "document.getElementById('galleryImage').complete && "
                        "document.getElementById('galleryImage').naturalWidth > 0"
                    )
        page.locator("#figureButton").click()
        assert page.locator("#imageDialog").is_visible()
        page.keyboard.press("Escape")
        page.locator('[data-doc="PROTOCOL.md"]').first.click()
        assert page.locator("#documentDialog").is_visible()
        assert "registered before" in page.locator("#documentContent").inner_text()
        page.locator("#documentSelect").select_option("evaluator/evaluation.json")
        assert json.loads(page.locator("#documentContent").inner_text()) == expected
        with page.expect_download() as source_download:
            page.locator("#downloadDocument").click()
        assert source_download.value.suggested_filename == "evaluator__evaluation.json"
        page.keyboard.press("Escape")
        with page.expect_download() as metrics_download:
            page.locator("#downloadMetrics").click()
        assert json.loads(Path(metrics_download.value.path()).read_text()) == expected
        page.locator("#showTruth").uncheck()
        page.locator("#figureArm").select_option("a-numpyro")
        page.locator("#figureStage").select_option("initial")
        page.locator("#figureKind").select_option("predictive")
        page.evaluate("window.scrollTo({top:0,behavior:'instant'})")
        page.screenshot(path=str(ROOT / "evaluator/html-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(ROOT / "evaluator/html-mobile.png"), full_page=True)
        assert not errors, errors
        assert not remote_requests, remote_requests
        browser.close()
    print("PASS: standalone/offline, four fit views, timeline keyboard controls, effort filters,")
    print(
        "18 embedded figures, evidence dialogs, exact JSON downloads, desktop/mobile, no JS errors."
    )


if __name__ == "__main__":
    main()
