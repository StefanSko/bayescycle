from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"


def test_explicit_documents_simulate_sample_and_check_recovery(page: Page, base_url: str) -> None:
    page.add_init_script(
        """const createObjectURL = URL.createObjectURL.bind(URL);
        window.__artifactBlobs = new Map();
        URL.createObjectURL = (blob) => {
            const url = createObjectURL(blob);
            window.__artifactBlobs.set(url, blob);
            return url;
        };"""
    )
    page.goto(f"{base_url}/site/")
    expect(page.locator("#generate-button")).to_be_attached()
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
    page.locator("#compile-button").click()
    expect(page.locator("#design-expr-x")).to_be_visible(timeout=120_000)
    expect(page.locator("#fixed-value-sigma")).to_have_value("")
    expect(page.locator("#generate-button")).to_be_disabled()
    page.locator("#design-expr-x").fill("linspace(-2, 2, 25)")
    expect(page.locator("#design-preview-x")).to_contain_text("shape [25]")
    page.locator("#fixed-value-alpha").fill("0.2")
    page.locator("#fixed-value-beta").fill("0.6")
    page.locator("#fixed-value-sigma").fill("0.8")
    expect(page.locator("#generate-button")).to_be_enabled()
    page.locator("#chains").fill("2")
    page.locator("#warmup").fill("8")
    page.locator("#draws").fill("12")
    page.locator("#generation-count").fill("1")
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-generated-datasets")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-generation-plan")).to_be_visible()
    expect(page.locator("#artifact-generation-run")).to_be_visible()
    forms_plan = page.locator("#artifact-generation-plan a").evaluate(
        "async (link) => JSON.parse(await window.__artifactBlobs.get(link.href).text())"
    )
    # Authoring through forms adds no provenance: the published plan carries
    # only authoritative fields, identical to direct-JSON authoring.
    assert "design_source" not in forms_plan
    expect(page.locator("#dataset-source-generated")).to_be_enabled()
    page.locator("#dataset-source-generated").check()
    expect(page.locator("#fit-button")).to_be_enabled()
    page.locator("#fit-button").click()
    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-recovery")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-generated-datasets")).to_be_visible()


def test_schema_names_that_match_object_prototype_keys_render_safely(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        """from bayeswire import Data, Observed, Param, model
from bayeswire.distributions import Normal

@model
class PrototypeNames:
    toString = Param(Normal(0.0, 1.0))
    constructor = Data.vector()
    y = Observed(Normal(toString + constructor, 1.0))
"""
    )
    page.locator("#compile-button").click()
    expect(page.locator("#design-expr-constructor")).to_have_value(
        "linspace(-2, 2, 25)", timeout=120_000
    )
    expect(page.locator("#fixed-value-toString")).to_have_value("0")
    assert '"constructor"' in page.locator("#design-data").input_value()
    assert '"toString"' in page.locator("#truth-data").input_value()
