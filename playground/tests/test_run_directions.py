from playwright.sync_api import Page, expect


def test_all_parameter_sources_use_one_native_generation_operation(
    page: Page, base_url: str
) -> None:
    page.add_init_script(
        """
        globalThis.__engineRequests = [];
        const NativeWorker = globalThis.Worker;
        globalThis.Worker = class RecordingWorker {
          constructor(url, options) {
            if (!String(url).includes("engine-worker.mjs")) {
              return new NativeWorker(url, options);
            }
            this.inner = new NativeWorker(url, options);
            this.onmessage = null;
            this.onerror = null;
            this.inner.onmessage = (event) => this.onmessage?.(event);
            this.inner.onerror = (event) => this.onerror?.(event);
          }
          postMessage(message) {
            if (message.request) globalThis.__engineRequests.push(structuredClone(message.request));
            this.inner.postMessage(message);
          }
          terminate() { this.inner.terminate(); }
        };
        """
    )
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#chains").fill("1")
    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("4")
    page.locator("#generation-count").fill("2")
    page.locator("#generation-seed").fill("123")
    page.locator("#inference-seed").fill("456")
    page.locator("#compile-button").click()

    expect(page.locator("#generate-button")).to_be_enabled(timeout=120_000)
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-generated-datasets")).to_be_visible(timeout=120_000)

    page.locator("#param-source-prior").check()
    expect(page.locator("#generate-button")).to_have_text(
        "Simulate 2 datasets from the model prior"
    )
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-generated-datasets")).to_be_visible(timeout=120_000)

    page.locator("#observed-data").fill('{"x":[-1,-0.5,0,0.5,1],"y":[-0.7,-0.1,0.5,1.1,1.7]}')
    page.locator("#fit-button").click()
    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#param-source-posterior")).to_be_enabled()
    page.locator("#param-source-posterior").check()
    expect(page.locator("#generate-button")).to_have_text("Simulate 2 datasets from the posterior")
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-generated-datasets")).to_be_visible(timeout=120_000)

    requests = page.evaluate("__engineRequests.filter((request) => request.command === 'generate')")
    assert [request["parameter_source"]["kind"] for request in requests] == [
        "fixed",
        "model-prior",
        "posterior",
    ]
    assert all(request["count"] == 2 and request["seed"] == 123 for request in requests)
    assert not page.evaluate(
        "__engineRequests.some((request) => "
        "['simulate', 'prior-predictive', 'posterior-predictive']"
        ".includes(request.command))"
    )

    page.locator("#observed-data").fill('{"x":[-1,0,1],"y":[-0.5,0.5,1.5]}')
    expect(page.locator("#param-source-fixed")).to_be_checked()
    expect(page.locator("#plan-summary")).to_contain_text("parameters: fixed values")
    expect(page.locator("#plan-summary")).not_to_contain_text("posterior from fit")
