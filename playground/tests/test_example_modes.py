from playwright.sync_api import Page, expect


def test_form_eligible_example_derives_design_forms_from_compiled_schema(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    menu = page.locator("#examples-menu")
    expect(menu.locator("option")).to_have_count(4)
    menu.select_option("eight-schools")
    expect(page.locator("#design-data")).to_have_value("")
    menu.select_option("linear-simulation")
    expect(page.locator("#design-data")).to_have_value("")
    expect(page.locator("#truth-data")).not_to_have_value("")
    page.locator("#compile-button").click()
    expect(page.locator("#design-slots")).to_be_visible(timeout=120_000)
    expect(page.locator("#design-json-field")).to_be_hidden()
    expect(page.locator("#design-expr-x")).to_have_value("linspace(-2, 2, 25)")
    expect(page.locator("#truth-json-field")).to_be_visible()


def test_pasted_form_eligible_model_with_empty_design_prefills_each_actual_slot(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n\n"
        "@model\n"
        "class TwoDesigns:\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    time = Data.vector()\n"
        "    dose = Data.vector(6)\n"
        "    y = Observed(Normal(beta * time + dose, 1.0))\n"
    )
    page.locator("#compile-button").click()
    expect(page.locator("#design-slots .slot-card")).to_have_count(2, timeout=120_000)
    expect(page.locator("#design-expr-time")).to_have_value("linspace(-2, 2, 25)")
    expect(page.locator("#design-expr-dose")).to_have_value("linspace(-2, 2, 6)")
    assert '"time"' in page.locator("#design-data").input_value()
    assert '"dose"' in page.locator("#design-data").input_value()
    assert '"x"' not in page.locator("#design-data").input_value()


def test_pasted_form_eligible_model_preserves_precompile_design_json(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n\n"
        "@model\n"
        "class ExplicitDesign:\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    x = Data.vector()\n"
        "    y = Observed(Normal(beta * x, 1.0))\n"
    )
    explicit = '{"x":[100.0,200.0,300.0]}'
    page.locator("#design-data").fill(explicit)
    page.locator("#compile-button").click()
    expect(page.locator("#design-json-field")).to_be_visible(timeout=120_000)
    expect(page.locator("#design-slots")).to_be_hidden()
    expect(page.locator("#design-data")).to_have_value(explicit)


def test_source_edit_rederives_forms_after_schema_forced_json_mode(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    non_eligible = (
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.constraints import Ordered\n"
        "from bayeswire.distributions import Normal, OrderedLogistic\n\n"
        "@model\n"
        "class FirstOrdinal:\n"
        "    n_cutpoints = Data.scalar()\n"
        "    x = Data.vector()\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())\n"
        "    y = Observed(OrderedLogistic(beta * x, cutpoints))\n"
    )
    eligible = (
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n\n"
        "@model\n"
        "class ThenLinear:\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    time = Data.vector()\n"
        "    y = Observed(Normal(beta * time, 1.0))\n"
    )
    page.locator("#model-source").fill(non_eligible)
    page.locator("#compile-button").click()
    expect(page.locator("#design-json-field")).to_be_visible(timeout=120_000)
    expect(page.locator("#design-json-toggle")).to_be_disabled()

    page.locator("#model-source").fill(eligible)
    page.locator("#compile-button").click()
    expect(page.locator("#design-slots")).to_be_visible(timeout=120_000)
    expect(page.locator("#design-json-field")).to_be_hidden()
    expect(page.locator("#design-expr-time")).to_have_value("linspace(-2, 2, 25)")
    assert '"n_cutpoints"' not in page.locator("#design-data").input_value()


def test_unknown_length_design_slots_get_no_runnable_placeholder(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    # A dimension-linked / unsized design vector has no static length, so it
    # cannot be scaffolded to a real shape; leaving [] would run generation
    # with zero rows. The design stays empty and generation disabled.
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.constraints import Ordered\n"
        "from bayeswire.distributions import Normal, OrderedLogistic\n\n"
        "@model\n"
        "class PastedOrdinal:\n"
        "    n_cutpoints = Data.scalar()\n"
        "    x = Data.vector()\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())\n"
        "    y = Observed(OrderedLogistic(beta * x, cutpoints))\n"
    )
    page.locator("#compile-button").click()
    # Wait for the compile to actually finish (#ir-hash appears) before checking
    # the post-compile design; an empty #design-data matches trivially in the
    # pre-compile state.
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)
    expect(page.locator("#design-json-field")).to_be_visible()
    expect(page.locator("#design-json-toggle")).to_be_disabled()
    expect(page.locator("#design-data")).to_have_value("")
    expect(page.locator("#generate-button")).to_be_disabled()


def test_non_eligible_placeholder_honors_exact_vector_lengths(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.constraints import Ordered\n"
        "from bayeswire.distributions import Normal, OrderedLogistic\n\n"
        "@model\n"
        "class FixedLenOrdinal:\n"
        "    n_cutpoints = Data.scalar()\n"
        "    x = Data.vector(3)\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())\n"
        "    y = Observed(OrderedLogistic(beta * x, cutpoints))\n"
    )
    page.locator("#compile-button").click()
    expect(page.locator("#design-json-field")).to_be_visible(timeout=120_000)
    # The exact-length x slot gets a length-3 float64 placeholder, not [] —
    # a shape-invalid design must not look runnable.
    expect(page.locator("#design-data")).to_have_value(
        '{"format":"bayescycle.data.json.v1","variables":{'
        '"n_cutpoints":{"dtype":"int64","shape":[],"values":[0]},'
        '"x":{"dtype":"float64","shape":[3],"values":[0,0,0]}}}'
    )


def test_unsupported_rank_design_slot_gets_no_runnable_placeholder(
    page: Page, base_url: str
) -> None:
    page.goto(f"{base_url}/site/")
    page.locator("#model-source").fill(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n\n"
        "@model\n"
        "class MatrixDesign:\n"
        "    x = Data.matrix()\n"
        "    beta = Param(Normal(0.0, 1.0))\n"
        "    y = Observed(Normal(beta * x[0, 0], 1.0))\n"
    )
    page.locator("#compile-button").click()
    # Wait for compile before asserting the empty design (empty matches the
    # pre-compile state trivially).
    expect(page.locator("#ir-hash")).to_be_visible(timeout=120_000)
    # A matrix slot has no schema-known shape, so no rank-1 placeholder is
    # emitted; the design stays empty and generation stays disabled.
    expect(page.locator("#design-data")).to_have_value("")
    expect(page.locator("#generate-button")).to_be_disabled()


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
    expect(page.locator("#design-slots")).to_be_visible(timeout=120_000)
    page.locator("#design-json-toggle").click()
    expect(page.locator("#design-json-field")).to_be_visible()
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
