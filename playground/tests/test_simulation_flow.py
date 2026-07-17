import json
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


def test_another_prior_simulates_composed_parameters_and_fits_original_model(
    page: Page, base_url: str
) -> None:
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
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#param-source-other-prior")).to_be_enabled(timeout=120_000)
    original_hash = page.locator("#ir-hash").text_content()
    page.locator("#param-source-other-prior").check()
    expect(page.locator("#prior-source")).to_be_visible()
    page.locator("#prior-source").fill(
        "@model\nclass SteepSlopes:\n"
        "    location = Param(Normal(3.0, 0.2))\n"
        "    beta = Param(Normal(location, 0.15))\n"
    )
    page.locator("#generation-count").fill("20")
    page.locator("#generation-seed").fill("17")
    expect(page.locator("#generate-button")).to_have_text(
        "Simulate 20 datasets from the composed prior"
    )
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-generated-datasets")).to_be_visible(timeout=120_000)

    generated = page.locator("#artifact-generated-datasets a").evaluate(
        "async (link) => (await window.__artifactBlobs.get(link.href).text())"
    )
    records = [json.loads(line) for line in generated.strip().splitlines()]
    beta_values = [
        record["parameters"]["variables"]["beta"]["values"][0] for record in records[1:-1]
    ]
    assert 2.8 < sum(beta_values) / len(beta_values) < 3.2
    assert all("location" in record["parameters"]["variables"] for record in records[1:-1])

    plan = page.locator("#artifact-generation-plan a").evaluate(
        "async (link) => JSON.parse(await window.__artifactBlobs.get(link.href).text())"
    )
    parameters = plan["distribution"]["parameters"]
    assert parameters["model_hash"] == plan["distribution"]["outcomes"]["model_hash"]
    assert (
        parameters["authored_provenance"]["claimed_source_model_hash"] == parameters["model_hash"]
    )
    assert parameters["authored_provenance"]["claimed_outcome_model_hash"] == original_hash

    page.locator("#chains").fill("1")
    page.locator("#warmup").fill("8")
    page.locator("#draws").fill("12")
    page.locator("#dataset-source-generated").check()
    page.locator("#fit-button").click()
    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-recovery")).to_be_visible(timeout=120_000)
    recovery = page.locator("#artifact-recovery a").evaluate(
        "async (link) => JSON.parse(await window.__artifactBlobs.get(link.href).text())"
    )
    assert "beta" in recovery["target_order"]
    assert "location" not in recovery["target_order"]
    generated_after_fit = page.locator("#artifact-generated-datasets a").evaluate(
        "async (link) => (await window.__artifactBlobs.get(link.href).text())"
    )
    assert generated_after_fit == generated
    expect(page.locator("#artifact-generation-model")).to_be_visible()
    model_hashes = page.evaluate(
        """async () => {
          const digest = async (selector) => {
            const link = document.querySelector(selector);
            const bytes = await window.__artifactBlobs.get(link.href).arrayBuffer();
            const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
            return 'sha256:' + [...hash]
              .map((value) => value.toString(16).padStart(2, '0')).join('');
          };
          return {
            conditioning: await digest('#artifact-model a'),
            generation: await digest('#artifact-generation-model a'),
          };
        }"""
    )
    assert model_hashes["conditioning"] == original_hash
    assert model_hashes["generation"] == parameters["model_hash"]


def test_failing_composed_generate_surfaces_engine_error_and_remains_actionable(
    page: Page, base_url: str
) -> None:
    page.add_init_script(
        """
        globalThis.__generateRequests = 0;
        const NativeWorker = globalThis.Worker;
        globalThis.Worker = class RecordingWorker {
          constructor(url, options) {
            if (!String(url).includes('engine-worker.mjs')) {
              return new NativeWorker(url, options);
            }
            this.inner = new NativeWorker(url, options);
            this.onmessage = null;
            this.onerror = null;
            this.inner.onmessage = (event) => this.onmessage?.(event);
            this.inner.onerror = (event) => this.onerror?.(event);
          }
          postMessage(message) {
            if (message.request?.command === 'generate') globalThis.__generateRequests += 1;
            this.inner.postMessage(message);
          }
          terminate() { this.inner.terminate(); }
        };
        """
    )
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#param-source-other-prior")).to_be_enabled(timeout=120_000)
    page.locator("#param-source-other-prior").check()
    page.locator("#prior-source").fill(
        "@model\nclass AlternativePrior:\n    beta = Param(Normal(3.0, 0.25))\n"
    )
    page.locator("#design-data").fill("{}")
    page.locator("#generation-count").fill("1")
    page.locator("#generate-button").click()
    expect(page.locator("#run-error")).not_to_be_empty(timeout=120_000)
    assert page.evaluate("globalThis.__generateRequests") == 1
    expect(page.locator("#run-error")).to_have_text(
        "DataShapeMismatch: generate design variable order must exactly match "
        "the model's declared data order"
    )
    expect(page.locator("#generate-button")).to_be_enabled()
    expect(page.locator("#cancel-run")).to_be_hidden()


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
