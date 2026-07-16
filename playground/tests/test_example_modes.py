from playwright.sync_api import Page, expect


def test_examples_load_raw_documents_in_json_escape_hatches(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    menu = page.locator("#examples-menu")
    expect(menu.locator("option")).to_have_count(4)
    menu.select_option("eight-schools")
    expect(page.locator("#design-data")).to_have_value("")
    menu.select_option("linear-simulation")
    expect(page.locator("#design-data")).not_to_have_value("")
    expect(page.locator("#truth-data")).not_to_have_value("")
    page.locator("#compile-button").click()
    expect(page.locator("#design-json-field")).to_be_visible(timeout=120_000)
    expect(page.locator("#truth-json-field")).to_be_visible()
    expect(page.locator("#design-slots")).to_be_hidden()


def test_scalar_design_slots_remain_in_the_json_escape_hatch(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("ordinal-simulation")
    expect(page.locator("#design-data")).not_to_have_value("")
    original = page.locator("#design-data").input_value()
    page.locator("#compile-button").click()
    expect(page.locator("#design-json-field")).to_be_visible(timeout=120_000)
    expect(page.locator("#design-json-toggle")).to_be_disabled()
    expect(page.locator("#design-data")).to_have_value(original)


def test_large_parameter_schema_stays_in_bounded_json_mode(page: Page, base_url: str) -> None:
    parameters = "\n".join(f"    p{index} = Param(Normal(0.0, 1.0))" for index in range(201))
    source = (
        "from bayeswire import Param, model\n"
        "from bayeswire.distributions import Normal\n\n"
        "@model\n"
        "class ManyParameters:\n"
        f"{parameters}\n"
    )
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(source)
    page.locator("#compile-button").click()
    expect(page.locator("#compile-status")).to_contain_text(
        "Compiled. 201 parameters", timeout=120_000
    )
    expect(page.locator("#truth-json-field")).to_be_visible()
    expect(page.locator("#truth-json-toggle")).to_be_disabled()
    expect(page.locator("#truth-json-toggle")).to_contain_text(
        "JSON required for more than 200 parameters"
    )
    expect(page.locator("#parameter-fields input")).to_have_count(0)


def test_entering_the_parameter_form_rewrites_a_stale_truth_document(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#truth-json-field")).to_be_visible(timeout=120_000)
    page.locator("#truth-data").fill('{"alpha": 1}')
    expect(page.locator("#generate-button")).to_be_enabled()

    page.locator("#truth-json-toggle").click()
    expect(page.locator("#fixed-value-alpha")).to_have_value("1")
    expect(page.locator("#fixed-value-sigma")).to_have_value("")
    # The hidden document now follows the form: sigma is missing, so the
    # stale {"alpha": 1} JSON must not remain simulatable.
    expect(page.locator("#generate-button")).to_be_disabled()

    page.locator("#fixed-value-beta").fill("0.6")
    page.locator("#fixed-value-sigma").fill("0.8")
    expect(page.locator("#generate-button")).to_be_enabled()


def test_large_design_schema_stays_in_bounded_json_mode(page: Page, base_url: str) -> None:
    slots = "\n".join(f"    x{index} = Data.vector()" for index in range(51))
    summed = " + ".join(f"x{index}" for index in range(51))
    source = (
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n\n"
        "@model\n"
        "class ManyDesignSlots:\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        f"{slots}\n"
        f"    y = Observed(Normal(beta * ({summed}), 1.0))\n"
    )
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(source)
    page.locator("#compile-button").click()
    expect(page.locator("#compile-status")).to_contain_text("51 design slots", timeout=120_000)
    expect(page.locator("#design-json-field")).to_be_visible()
    expect(page.locator("#design-json-toggle")).to_be_disabled()
    expect(page.locator("#design-json-toggle")).to_contain_text(
        "JSON required for more than 50 design slots"
    )
    expect(page.locator("#design-slots .slot-card")).to_have_count(0)


def test_zero_product_canonical_shape_stays_in_json_mode(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#design-json-field")).to_be_visible(timeout=120_000)
    page.locator("#design-data").fill(
        '{"format":"bayescycle.data.json.v1","variables":'
        '{"x":{"dtype":"float64","shape":[1000000000,0],"values":[]}}}'
    )
    page.locator("#design-json-toggle").click()
    expect(page.locator("#design-json-field")).to_be_visible()
    expect(page.locator("#authoring-error")).to_contain_text(
        "document shape projection exceeds 100000 form values"
    )


def test_proto_named_form_field_preserves_its_canonical_document(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n\n"
        "@model\n"
        "class ProtoDesign:\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    __proto__ = Data.vector()\n"
        "    y = Observed(Normal(beta * __proto__, 1.0))\n"
    )
    page.locator("#compile-button").click()
    expect(page.locator("#design-slots")).to_be_visible(timeout=120_000)
    page.locator("#design-expr-__proto__").fill("[1, 2]")
    assert '"__proto__"' in page.locator("#design-data").input_value()
    expect(page.locator("#design-expr-__proto__")).to_be_visible()


def test_form_aggregate_scalar_cap_surfaces_before_materialization(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n\n"
        "@model\n"
        "class BoundedDesign:\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    x = Data.vector()\n"
        "    z = Data.vector()\n"
        "    y = Observed(Normal(beta * x + z, 1.0))\n"
    )
    page.locator("#compile-button").click()
    expect(page.locator("#design-slots")).to_be_visible(timeout=120_000)
    page.locator("#design-expr-x").fill("repeat([1], 50001)")
    page.locator("#design-expr-z").fill("repeat([1], 50001)")
    expect(page.locator("#authoring-error")).to_contain_text("maximum scalar count of 100000")
    expect(page.locator("#generate-button")).to_be_disabled()


def test_exact_length_design_slots_enforce_their_shape(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n"
        "\n"
        "@model\n"
        "class FixedDesign:\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    x = Data.vector(3)\n"
        "    w = Data.vector(1)\n"
        "    y = Observed(Normal(beta * x + w[0], 1.0))\n"
    )
    page.locator("#compile-button").click()
    expect(page.locator("#design-slots")).to_be_visible(timeout=120_000)
    expect(page.locator("#design-expr-x")).to_have_value("linspace(-2, 2, 3)")
    expect(page.locator("#design-expr-w")).to_have_value("[0]")
    expect(page.locator("#generate-button")).to_be_enabled()

    page.locator("#design-expr-x").fill("linspace(-2, 2, 25)")
    expect(page.locator("#design-preview-x")).to_contain_text("needs exactly 3 values")
    expect(page.locator("#generate-button")).to_be_disabled()

    page.locator("#design-expr-x").fill("linspace(-2, 2, 3)")
    expect(page.locator("#generate-button")).to_be_enabled()

    # A cleared numeric field is invalid input, not an implicit zero.
    page.locator("#generation-seed").fill("")
    expect(page.locator("#generate-button")).to_be_disabled()
    page.locator("#generation-seed").fill("0")
    expect(page.locator("#generate-button")).to_be_enabled()
