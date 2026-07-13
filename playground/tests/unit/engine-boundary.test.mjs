import { WorkerEngine } from "/site/src/engine/executor.mjs";

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
}];
