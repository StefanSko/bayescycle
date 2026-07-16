import * as compilerModule from "/site/src/compile/index.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const MODEL_SCHEMA = {
  schema_format: "bayescycle.playground.model-schema.v0",
  parameters: [],
  data: [],
  observed: [],
};

class FakeWorker {
  constructor(onPost = () => {}) {
    this.listeners = new Map();
    this.messages = [];
    this.terminated = false;
    this.onPost = onPost;
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((entry) => entry !== listener));
  }

  postMessage(message) {
    this.messages.push(message);
    this.onPost(message, this);
  }

  terminate() {
    this.terminated = true;
  }

  emit(type, data = {}) {
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener({ data, message: data.message ?? "worker failed" });
  }
}

function client(options) {
  const CompilerClient = compilerModule.CompilerClient;
  assert(typeof CompilerClient === "function", "CompilerClient is not exported");
  return new CompilerClient(options);
}

function readyThen(response) {
  return () => {
    const worker = new FakeWorker((request, active) => queueMicrotask(() => active.emit("message", response(request))));
    queueMicrotask(() => worker.emit("message", { type: "ready", protocol: 1 }));
    return worker;
  };
}

async function rejectedAfterTermination(promise, worker, expected) {
  let error;
  try {
    await promise.catch((reason) => {
      assert(worker.terminated, "compile rejected before its worker was terminated");
      throw reason;
    });
  } catch (reason) {
    error = reason;
  }
  assert(String(error).includes(expected), `expected ${expected} rejection, got ${String(error)}`);
}

export default [
  {
    name: "runtime owns the compiler adapter",
    fn: async () => {
      const calls = [];
      const adapter = { compile: async (source, options) => { calls.push({ source, options }); return { ok: false, message: "declined" }; } };
      const runtimeModule = await import("/site/src/runtime/browser-runtime.mjs");
      const runtime = new runtimeModule.BrowserRuntime({ execute() {} }, adapter);
      const result = await runtime.compile("model source", { timeoutMs: 7 });
      assert(result.message === "declined", "runtime did not return compiler result");
      assert(calls.length === 1 && calls[0].source === "model source", "runtime did not delegate compilation");
    },
  },
  {
    name: "each compile owns a fresh source-only worker request",
    fn: async () => {
      const workers = [];
      const factory = readyThen((request) => ({ type: "compile-error", id: request.id, exceptionType: "ValueError", message: "no", traceback: "no" }));
      const compiler = client({ workerFactory: () => { const worker = factory(); workers.push(worker); return worker; } });
      await compiler.compile("first");
      await compiler.compile("second");
      assert(workers.length === 2 && workers[0] !== workers[1], "compile attempts reused a worker");
      assert(workers.every((worker) => worker.terminated), "a compile worker survived settlement");
      for (const [index, worker] of workers.entries()) {
        const request = worker.messages[0];
        assert(JSON.stringify(Object.keys(request).sort()) === JSON.stringify(["id", "protocol", "source", "type"]), `request ${index} was not source-only: ${Object.keys(request)}`);
      }
    },
  },
  {
    name: "worker terminates before success and declaration failure settle",
    fn: async () => {
      for (const response of [
        (request) => ({ type: "compiled", id: request.id, irBytes: new TextEncoder().encode("{}").buffer, modelSchema: MODEL_SCHEMA, irHash: "untrusted" }),
        (request) => ({ type: "compile-error", id: request.id, exceptionType: "ValueError", message: "bad model", traceback: "trace" }),
      ]) {
        let worker;
        const compiler = client({ workerFactory: () => { worker = readyThen(response)(); return worker; } });
        await compiler.compile("source").then((result) => {
          assert(worker.terminated, "compile settled before termination");
          if (result.ok) {
            assert(result.irHash === "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a", `worker digest was trusted: ${result.irHash}`);
          }
        });
      }
    },
  },
  {
    name: "worker terminates on startup failure malformed response and worker error",
    fn: async () => {
      const startupWorker = new FakeWorker();
      const startup = client({ workerFactory: () => startupWorker, startupTimeoutMs: 5 });
      await rejectedAfterTermination(startup.compile("source"), startupWorker, "startup");

      let malformedWorker;
      const malformed = client({ workerFactory: () => {
        malformedWorker = readyThen((request) => ({ type: "compiled", id: request.id, irBytes: "not bytes" }))();
        return malformedWorker;
      } });
      await rejectedAfterTermination(malformed.compile("source"), malformedWorker, "malformed");

      let malformedSchemaWorker;
      const malformedSchema = client({ workerFactory: () => {
        malformedSchemaWorker = readyThen((request) => ({
          type: "compiled", id: request.id,
          irBytes: new TextEncoder().encode("{}").buffer,
          modelSchema: { ...MODEL_SCHEMA, extra: true },
        }))();
        return malformedSchemaWorker;
      } });
      await rejectedAfterTermination(
        malformedSchema.compile("source"), malformedSchemaWorker, "malformed model schema",
      );
      let dtypeError = "";
      try {
        compilerModule.validateModelSchema({
          ...MODEL_SCHEMA,
          data: [{ name: "x", dtype: "float128", kind: "vector", length: null }],
        });
      } catch (error) {
        dtypeError = String(error);
      }
      assert(dtypeError.includes("dtype is unsupported"), "unknown schema dtype was accepted");

      compilerModule.validateModelSchema({
        ...MODEL_SCHEMA,
        data: [{ name: "x", dtype: "float64", kind: "vector", length: 0 }],
      });
      for (const length of [-1, 1.5, "3"]) {
        let lengthError = "";
        try {
          compilerModule.validateModelSchema({
            ...MODEL_SCHEMA,
            data: [{ name: "x", dtype: "float64", kind: "vector", length }],
          });
        } catch (error) {
          lengthError = String(error);
        }
        assert(
          lengthError.includes("length must be a non-negative integer"),
          `schema length ${JSON.stringify(length)} was accepted`,
        );
      }

      let errorWorker;
      const failing = client({ workerFactory: () => {
        errorWorker = new FakeWorker(() => queueMicrotask(() => errorWorker.emit("error", { message: "boom" })));
        queueMicrotask(() => errorWorker.emit("message", { type: "ready", protocol: 1 }));
        return errorWorker;
      } });
      await rejectedAfterTermination(failing.compile("source"), errorWorker, "boom");

      const thrown = client({ workerFactory: () => { throw new Error("constructor failed"); } });
      let message = "";
      try { await thrown.compile("source"); } catch (error) { message = String(error); }
      assert(message.includes("constructor failed"), `worker constructor failure disappeared: ${message}`);
    },
  },
  {
    name: "oversized compiler output is bounded before use",
    fn: async () => {
      let worker;
      const compiler = client({
        workerFactory: () => {
          worker = readyThen((request) => ({ type: "compiled", id: request.id, irBytes: new Uint8Array([1, 2]).buffer, modelSchema: MODEL_SCHEMA }))();
          return worker;
        },
        maxOutputBytes: 1,
      });
      await rejectedAfterTermination(compiler.compile("source"), worker, "exceeds 1 byte");
    },
  },
  {
    name: "worker terminates on compile timeout and cancellation",
    fn: async () => {
      let timeoutWorker;
      const timeout = client({ workerFactory: () => {
        timeoutWorker = new FakeWorker();
        queueMicrotask(() => timeoutWorker.emit("message", { type: "ready", protocol: 1 }));
        return timeoutWorker;
      }, timeoutMs: 5 });
      await rejectedAfterTermination(timeout.compile("while True: pass"), timeoutWorker, "timed out");

      let cancelledWorker;
      const cancelled = client({ workerFactory: () => {
        cancelledWorker = new FakeWorker();
        queueMicrotask(() => cancelledWorker.emit("message", { type: "ready", protocol: 1 }));
        return cancelledWorker;
      } });
      const controller = new AbortController();
      const pending = cancelled.compile("source", { signal: controller.signal });
      controller.abort();
      await rejectedAfterTermination(pending, cancelledWorker, "cancelled");
    },
  },
];
