from playwright.sync_api import Page, expect


def test_composed_actions_availability_fallback_and_separate_seeds(
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

    expect(page.locator("#generate-button")).to_have_text("Generate at fixed values")
    expect(page.locator("#fixed-values-field")).to_be_visible()
    page.locator("#param-source-prior").check()
    expect(page.locator("#generate-button")).to_have_text("Generate from model prior")
    expect(page.locator("#fixed-values-field")).to_be_hidden()
    page.locator("#param-source-fixed").check()
    expect(page.locator("#fit-button")).to_have_text("Fit observed data")
    expect(page.locator("#param-source-posterior")).to_be_disabled()

    page.locator("#chains").fill("1")
    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("4")
    page.locator("#generation-seed").fill("123")
    page.locator("#inference-seed").fill("456")
    page.locator("#compile-button").click()
    expect(page.locator("#generate-button")).to_be_enabled(timeout=120_000)
    page.locator("#generate-button").click()
    expect(page.locator("#artifact-simulated")).to_be_visible(timeout=120_000)
    assert (
        page.evaluate("__engineRequests.find((request) => request.command === 'simulate').seed")
        == 123
    )

    expect(page.locator("#dataset-source-generated")).to_be_enabled()
    page.locator("#warmup").fill("5")
    expect(page.locator("#artifact-simulated")).to_be_visible()
    expect(page.locator("#dataset-source-generated")).to_be_enabled()
    page.locator("#dataset-source-generated").check()
    expect(page.locator("#fit-button")).to_have_text("Fit generated dataset")
    page.locator("#fit-button").click()
    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    assert (
        page.evaluate("__engineRequests.find((request) => request.command === 'sample').seed")
        == 456
    )

    expect(page.locator("#param-source-posterior")).to_be_enabled()
    page.locator("#param-source-posterior").check()
    expect(page.locator("#generate-button")).to_have_text("Generate from posterior")

    page.locator("#generation-seed").fill("124")
    expect(page.locator("#artifact-posterior")).to_be_visible()
    expect(page.locator("#param-source-posterior")).to_be_enabled()
    expect(page.locator("#param-source-posterior")).to_be_checked()
    expect(page.locator("#generate-button")).to_have_text("Generate from posterior")
    expect(page.locator("#artifact-simulated")).to_be_hidden()
    expect(page.locator("#dataset-source-generated")).to_be_disabled()
    expect(page.locator("#dataset-source-observed")).to_be_checked()
    expect(page.locator("#fit-button")).to_have_text("Fit observed data")

    page.locator("#warmup").fill("6")
    expect(page.locator("#artifact-posterior")).to_be_hidden()
    expect(page.locator("#param-source-posterior")).to_be_disabled()
    expect(page.locator("#param-source-fixed")).to_be_checked()
    expect(page.locator("#generate-button")).to_have_text("Generate at fixed values")
    expect(page.locator("#fit-button")).to_have_text("Fit observed data")
