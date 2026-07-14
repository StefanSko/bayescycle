from playwright.sync_api import Page, expect


def test_recovery_summary_remains_visible_when_diagnostics_fail(page: Page, base_url: str) -> None:
    page.add_init_script(
        """
        const NativeWorker = globalThis.Worker;
        globalThis.Worker = class EngineFailureWorker {
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
            if (message.request?.command === "diagnose") {
              queueMicrotask(() => this.onmessage?.({ data: {
                type: "error", id: message.id, chainId: message.chainId,
                error: {
                  error_format: "v0-provisional",
                  error: "InjectedFailure",
                  message: "diagnostics unavailable",
                },
              }}));
              return;
            }
            this.inner.postMessage(message);
          }
          terminate() { this.inner.terminate(); }
        };
        """
    )
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("linear-simulation")
    page.locator("#compile-button").click()
    expect(page.locator("#generate-button")).to_be_enabled(timeout=120_000)
    page.locator("#chains").fill("1")
    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("4")
    page.locator("#generate-button").click()
    expect(page.locator("#dataset-source-generated")).to_be_enabled(timeout=120_000)
    page.locator("#dataset-source-generated").check()
    page.locator("#fit-button").click()
    expect(page.locator("#artifact-recovery")).to_be_visible(timeout=120_000)
    expect(page.locator("#run-error")).to_contain_text("diagnostics unavailable")
    expect(page.locator("#recovery-summary")).to_be_visible()

    page.locator("#inference-seed").fill("201")
    expect(page.locator("#artifact-posterior")).to_be_visible()
    expect(page.locator("#run-error")).to_be_visible()
    expect(page.locator("#run-error")).to_contain_text("diagnostics unavailable")
