import { WorkerEngine } from "/site/src/engine/executor.mjs";
import { sample } from "/site/src/engine/verbs.mjs";

function assert(condition, message) { if (!condition) throw new Error(message); }

class FakeWorker {
  constructor(response) {
    this.response = response;
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
  }
  postMessage(message) {
    queueMicrotask(() => this.onmessage?.({ data: this.response(message) }));
  }
  terminate() { this.terminated = true; }
}

async function rejectsBounded(response) {
  let worker;
  const engine = new WorkerEngine("test", "wasm", "metadata", () => {
    worker = new FakeWorker(response);
    return worker;
  });
  const outcome = await Promise.race([
    engine.execute({ command: "diagnose", fit: "fit" }).then(
      () => "resolved",
      (error) => String(error),
    ),
    new Promise((resolve) => setTimeout(() => resolve("hung"), 100)),
  ]);
  assert(outcome !== "resolved" && outcome !== "hung", `malformed response outcome: ${outcome}`);
  assert(outcome.includes("MalformedEngineResponse"), `missing typed error: ${outcome}`);
  assert(worker.terminated, "engine worker survived malformed response");
}

export default [{
  name: "engine worker rejects malformed and unknown responses",
  fn: async () => {
    await rejectsBounded(() => null);
    await rejectsBounded((request) => ({ type: "bogus", id: request.id }));
    await rejectsBounded((request) => ({
      type: "future-result",
      id: request.id,
      chainId: 0,
      rawBytes: new Uint8Array([1]),
    }));
    await rejectsBounded((request) => ({ type: "result", id: request.id, rawBytes: "not bytes" }));
    await rejectsBounded((request) => ({ type: "error", id: request.id, error: { message: "missing kind" } }));
    await rejectsBounded((request) => ({ type: "batch", id: request.id, chainId: 0, draws: "not draws" }));
  },
}, {
  name: "first chain failure and run abort terminate sibling workers",
  fn: async () => {
    const workersAfterFailure = [];
    const failingEngine = new WorkerEngine("test", "wasm", "metadata", () => {
      const worker = new FakeWorker((message) => message.chainId === 0 ? {
        type: "error",
        id: message.id,
        chainId: 0,
        error: {
          error_format: "v0-provisional",
          error: "ChainFailure",
          message: "first chain failed",
        },
      } : { type: "started", id: "stale", chainId: message.chainId });
      workersAfterFailure.push(worker);
      return worker;
    });
    const failed = await sample({
      model: {}, data: {}, settings: {}, seed: 0, chains: 3,
      executor: failingEngine,
    });
    assert(!failed.ok && failed.error.error === "ChainFailure", "first chain failure disappeared");
    assert(workersAfterFailure.length === 3, "not all sibling workers launched");
    assert(workersAfterFailure.every((worker) => worker.terminated), "a failed run retained sibling workers");

    const workersAfterAbort = [];
    const pendingEngine = new WorkerEngine("test", "wasm", "metadata", () => {
      const worker = new FakeWorker(() => ({ type: "started", id: "stale", chainId: 0 }));
      workersAfterAbort.push(worker);
      return worker;
    });
    const controller = new AbortController();
    const pending = sample({
      model: {}, data: {}, settings: {}, seed: 0, chains: 3,
      executor: pendingEngine, signal: controller.signal,
    });
    controller.abort();
    const aborted = await pending;
    assert(!aborted.ok && aborted.error.error === "Cancelled", "run abort was not typed");
    assert(workersAfterAbort.length === 3, "not all abort siblings launched");
    assert(workersAfterAbort.every((worker) => worker.terminated), "an aborted run retained sibling workers");
  },
}, {
  name: "aborting an engine execution terminates its worker with a typed cancellation",
  fn: async () => {
    let worker;
    const engine = new WorkerEngine("test", "wasm", "metadata", () => {
      worker = new FakeWorker(() => ({ type: "started", id: "stale", chainId: 0 }));
      return worker;
    });
    const controller = new AbortController();
    const pending = engine.execute(
      { command: "diagnose", fit: "fit" },
      { signal: controller.signal },
    );
    controller.abort();
    let error;
    try { await pending; } catch (reason) { error = reason; }
    assert(error?.error === "Cancelled", `cancellation was not typed: ${String(error)}`);
    assert(worker.terminated, "aborted engine worker survived");
  },
}];
