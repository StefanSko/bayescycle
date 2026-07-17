from playwright.sync_api import Page, expect


def test_oversized_posterior_is_actionable_and_normal_retry_renders(
    page: Page, base_url: str
) -> None:
    page.add_init_script(
        """
        const NativeWorker = globalThis.Worker;
        let rejectNextSample = true;
        globalThis.__posteriorLimitWorkerTerminated = false;
        globalThis.Worker = class PosteriorLimitWorker {
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
            if (rejectNextSample && message.request?.command === "sample") {
              rejectNextSample = false;
              this.inner.terminate();
              this.inner = null;
              class OversizedBytes extends Uint8Array {
                get byteLength() { return 64 * 1024 * 1024 + 1; }
              }
              queueMicrotask(() => this.onmessage?.({ data: {
                type: "result",
                id: message.id,
                chainId: message.chainId,
                rawBytes: new OversizedBytes(0),
              }}));
              return;
            }
            this.inner.postMessage(message);
          }
          terminate() {
            globalThis.__posteriorLimitWorkerTerminated = true;
            this.inner?.terminate();
            this.inner = null;
          }
        };
        """
    )
    page.goto(f"{base_url}/site/")
    page.locator("#examples-menu").select_option("eight-schools")
    page.locator("#compile-button").click()
    expect(page.locator("#fit-button")).to_be_enabled(timeout=120_000)

    page.locator("#chains").fill("1")
    page.locator("#warmup").fill("4")
    page.locator("#draws").fill("4")
    page.locator("#max-treedepth").fill("4")
    page.locator("#fit-button").click()

    expect(page.locator("#run-error")).to_have_text(
        "Posterior exceeds the 64 MiB browser limit; reduce parameters or draws.",
        timeout=5_000,
    )
    expect(page.locator("#cancel-run")).to_be_hidden()
    expect(page.locator("#fit-button")).to_be_enabled()
    expect(page.locator("#artifact-posterior")).to_be_hidden()
    expect(page.locator("#artifact-diagnostics")).to_be_hidden()
    expect(page.locator("#artifact-recovery")).to_be_hidden()
    assert page.evaluate("globalThis.__posteriorLimitWorkerTerminated") is True

    page.locator("#fit-button").click()
    expect(page.locator("#artifact-posterior")).to_be_visible(timeout=120_000)
    expect(page.locator("#artifact-diagnostics")).to_be_visible(timeout=120_000)
    expect(page.locator("#plots")).to_be_visible(timeout=5_000)
