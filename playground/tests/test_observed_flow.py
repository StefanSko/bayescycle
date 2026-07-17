from pathlib import Path

from playwright.sync_api import Page, expect

FIXTURES = Path(__file__).parent / "fixtures"


def test_observed_ready_hint_requires_valid_settings_and_required_variables(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    compact = (
        '{"n_schools":8,'
        '"sigma":[15.0,10.0,16.0,11.0,9.0,11.0,10.0,18.0],'
        '"y":[28.0,8.0,-3.0,7.0,-1.0,1.0,18.0,12.0]}\n'
    )
    expect(page.locator("#observed-data")).to_have_value(compact)
    expect(page.locator("#observed-ready-hint")).to_be_hidden()

    page.locator("#compile-button").click()
    expect(page.locator("#observed-ready-hint")).to_be_visible(timeout=120_000)
    expect(page.locator("#observed-ready-hint")).to_have_text(
        "No simulate run needed — fit this observed data directly."
    )

    page.locator("#chains").fill("0")
    expect(page.locator("#observed-ready-hint")).to_be_hidden()
    page.locator("#chains").fill("4")
    expect(page.locator("#observed-ready-hint")).to_be_visible()

    page.locator("#observed-data").fill("{}")
    expect(page.locator("#observed-ready-hint")).to_be_hidden()
    page.locator("#observed-data").fill('{"n_schools":8,"sigma":[15,10,16,11,9,11,10,18]}')
    expect(page.locator("#observed-ready-hint")).to_be_hidden()
    page.locator("#observed-data").fill(compact)
    expect(page.locator("#observed-ready-hint")).to_be_visible()

    page.locator("#observed-data").fill("{")
    expect(page.locator("#observed-ready-hint")).to_be_hidden()


def test_observed_model_to_artifacts(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    source = (FIXTURES / "corpus" / "eight_schools_non_centered.py").read_text()
    data = (FIXTURES / "engine" / "eight_schools_non_centered" / "data.json").read_text()

    page.locator("#model-source").fill(source)
    page.locator("#observed-data").fill(data)
    expect(page.locator("#design-data")).to_have_value("")
    expect(page.locator("#ir-hash")).to_be_hidden()

    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_have_text(
        "sha256:6cb101cf5159bddcbe10650a87a8763054a462da0f4374b8b61a2a1a861695dc",
        timeout=120_000,
    )

    page.locator("#chains").fill("2")
    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("4")
    expect(page.locator("#fit-button")).to_be_enabled()
    page.locator("#fit-button").click()
    expect(page.locator("#run-status")).to_have_text("Fitting is running…")
    expect(page.locator("#observed-ready-hint")).to_be_hidden()
    expect(page.locator("#compile-button")).to_be_disabled()

    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-diagnostics")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-model")).to_be_visible()
    expect(page.locator("#artifact-data")).to_be_visible()
    expect(page.locator("#progress")).to_contain_text("chain 1")
    expect(page.locator("#run-error")).to_be_hidden()

    expect(page.locator("#param-source-posterior")).to_be_enabled()
    expect(page.locator("#design-json-toggle")).to_be_disabled()
    expect(page.locator("#design-data")).to_be_visible()
    page.locator("#design-data").fill('{"n_schools":8,"sigma":[15,10,16,11,9,11,10,18]}')
    page.locator("#generation-count").fill("2")
    page.locator("#param-source-posterior").check()
    expect(page.locator("#generate-button")).to_have_text("Simulate 2 datasets from the posterior")
    expect(page.locator("#generate-button")).to_be_enabled()
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-generated-datasets")).to_be_visible(timeout=120_000)

    page.locator("#generation-seed").fill("1")
    expect(page.locator("#artifact-generated-datasets")).to_be_hidden()
    expect(page.locator("#artifact-posterior")).to_be_visible()
    expect(page.locator("#param-source-posterior")).to_be_enabled()
    expect(page.locator("#param-source-posterior")).to_be_checked()

    page.locator("#examples-menu").select_option("linear-simulation")
    expect(page.locator("#progress")).to_be_empty()


def test_ready_hint_validates_known_fixed_lengths(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n\n"
        "@model\n"
        "class FixedObs:\n"
        "    x = Data.vector(3)\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    y = Observed(Normal(beta * x, 1.0))\n"
    )
    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)
    # x is a fixed-length vector(3); a 2-element x is a known shape mismatch,
    # so the cue must stay hidden even though both names are present.
    page.locator("#observed-data").fill('{"x":[1,2],"y":[0,0]}')
    expect(page.locator("#observed-ready-hint")).to_be_hidden()
    # A schema-correct document shows the cue.
    page.locator("#observed-data").fill('{"x":[1,2,3],"y":[0,0,0]}')
    expect(page.locator("#observed-ready-hint")).to_be_visible()


def test_ready_hint_validates_known_integer_slot_dtypes(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.constraints import Ordered\n"
        "from bayeswire.distributions import Normal, OrderedLogistic\n\n"
        "@model\n"
        "class Ord:\n"
        "    n_cutpoints = Data.scalar()\n"
        "    x = Data.vector(3)\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())\n"
        "    y = Observed(OrderedLogistic(beta * x, cutpoints))\n"
    )
    page.locator("#compile-button").click()
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)
    # n_cutpoints is an integer slot; a fractional value is a known dtype
    # mismatch, so the cue stays hidden even with correct names/shapes.
    page.locator("#observed-data").fill('{"n_cutpoints":2.5,"x":[1,2,3],"y":[0,1,2]}')
    expect(page.locator("#observed-ready-hint")).to_be_hidden()
    # Integer n_cutpoints and integer x (accepted for the float slot) bind.
    page.locator("#observed-data").fill('{"n_cutpoints":2,"x":[1,2,3],"y":[0,1,2]}')
    expect(page.locator("#observed-ready-hint")).to_be_visible()
