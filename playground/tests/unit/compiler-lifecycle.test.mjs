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

      const scenarioCalls = [];
      const scenarioRuntime = new runtimeModule.BrowserRuntime({ execute() {} }, {
        compileScenario: async (source, priorSource, options) => {
          scenarioCalls.push({ source, priorSource, options });
          return { ok: false, message: "scenario declined" };
        },
      });
      const scenario = await scenarioRuntime.compileScenario("main", "prior", { timeoutMs: 9 });
      assert(scenario.message === "scenario declined", "runtime lost scenario result");
      assert(
        scenarioCalls.length === 1 && scenarioCalls[0].source === "main" &&
          scenarioCalls[0].priorSource === "prior",
        "runtime did not delegate scenario compilation",
      );
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
    name: "each scenario compile owns a fresh bounded worker request",
    fn: async () => {
      const workers = [];
      const factory = readyThen((request) => ({
        type: "compile-error", id: request.id, exceptionType: "ValueError",
        message: "no", traceback: "no",
      }));
      const compiler = client({
        workerFactory: () => {
          const worker = factory();
          workers.push(worker);
          return worker;
        },
      });
      await compiler.compileScenario("main one", "prior one");
      await compiler.compileScenario("main two", "prior two");
      assert(workers.length === 2 && workers.every((worker) => worker.terminated),
        "scenario compiles reused or retained a worker");
      for (const [index, worker] of workers.entries()) {
        const request = worker.messages[0];
        assert(request.mode === "with-prior", `scenario ${index} lost its mode`);
        assert(request.source === `main ${index === 0 ? "one" : "two"}`,
          `scenario ${index} lost main source`);
        assert(request.priorSource === `prior ${index === 0 ? "one" : "two"}`,
          `scenario ${index} lost prior source`);
        assert(!("design" in request) && !("truth" in request),
          `scenario ${index} leaked project data`);
      }

      let constructions = 0;
      const bounded = client({ workerFactory: () => { constructions += 1; return new FakeWorker(); } });
      let snippetMessage = "";
      try {
        await bounded.compileScenario("main", "é".repeat(
          Math.floor(compilerModule.MAX_MODEL_SOURCE_BYTES / 2) + 1,
        ));
      } catch (error) { snippetMessage = String(error); }
      assert(snippetMessage.includes("Prior-only source"), `snippet cap was unclear: ${snippetMessage}`);
      let combinedMessage = "";
      try {
        await bounded.compileScenario(
          "m".repeat(compilerModule.MAX_MODEL_SOURCE_BYTES / 2 + 1),
          "p".repeat(compilerModule.MAX_MODEL_SOURCE_BYTES / 2),
        );
      } catch (error) { combinedMessage = String(error); }
      assert(combinedMessage.includes("together"), `combined cap was unclear: ${combinedMessage}`);
      assert(constructions === 0, "oversized scenario constructed a worker");
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
    name: "oversized model source is rejected before worker construction",
    fn: async () => {
      let boundaryWorker;
      const boundaryCompiler = client({
        workerFactory: () => {
          boundaryWorker = readyThen((request) => ({
            type: "compile-error", id: request.id,
            exceptionType: "ValueError", message: "bounded", traceback: "bounded",
          }))();
          return boundaryWorker;
        },
      });
      const boundaryResult = await boundaryCompiler.compile(
        "x".repeat(compilerModule.MAX_MODEL_SOURCE_BYTES),
      );
      assert(!boundaryResult.ok, "exact source byte cap did not reach the worker");
      assert(boundaryWorker.terminated, "boundary source worker survived settlement");

      let workerConstructions = 0;
      const compiler = client({
        workerFactory: () => {
          workerConstructions += 1;
          return new FakeWorker();
        },
      });
      const source = "é".repeat(Math.floor(compilerModule.MAX_MODEL_SOURCE_BYTES / 2) + 1);
      let message = "";
      try { await compiler.compile(source); } catch (error) { message = String(error); }
      assert(message.includes("UTF-8 size"), `source cap error was unclear: ${message}`);
      assert(message.includes(String(compilerModule.MAX_MODEL_SOURCE_BYTES)), `source cap value missing: ${message}`);
      assert(workerConstructions === 0, "oversized source constructed a worker");
    },
  },
  {
    name: "model schema cardinality and all text boundaries are enforced",
    fn: () => {
      const parameter = (name, overrides = {}) => ({
        name,
        prior: "Normal(0, 1)",
        constraint: null,
        shape: [],
        default: null,
        ...overrides,
      });
      compilerModule.validateModelSchema({
        ...MODEL_SCHEMA,
        parameters: Array.from(
          { length: compilerModule.MAX_MODEL_SCHEMA_ENTRIES },
          (_, index) => parameter(`p${index}`),
        ),
      });
      let cardinalityMessage = "";
      try {
        compilerModule.validateModelSchema({
          ...MODEL_SCHEMA,
          parameters: Array.from(
            { length: compilerModule.MAX_MODEL_SCHEMA_ENTRIES + 1 },
            (_, index) => parameter(`p${index}`),
          ),
        });
      } catch (error) { cardinalityMessage = String(error); }
      assert(cardinalityMessage.includes("malformed model schema"), `wrong cardinality domain: ${cardinalityMessage}`);
      assert(cardinalityMessage.includes(String(compilerModule.MAX_MODEL_SCHEMA_ENTRIES)), `cardinality cap missing: ${cardinalityMessage}`);

      const atCap = "x".repeat(compilerModule.MAX_MODEL_SCHEMA_TEXT_CHARACTERS);
      compilerModule.validateModelSchema({
        ...MODEL_SCHEMA,
        parameters: [parameter(atCap, {
          prior: atCap, constraint: atCap, shape: [atCap],
        })],
      });
      for (const [field, value] of [
        ["name", "n".repeat(compilerModule.MAX_MODEL_SCHEMA_TEXT_CHARACTERS + 1)],
        ["prior", "p".repeat(compilerModule.MAX_MODEL_SCHEMA_TEXT_CHARACTERS + 1)],
        ["constraint", "c".repeat(compilerModule.MAX_MODEL_SCHEMA_TEXT_CHARACTERS + 1)],
      ]) {
        let message = "";
        try {
          compilerModule.validateModelSchema({
            ...MODEL_SCHEMA,
            parameters: [parameter("short", { [field]: value })],
          });
        } catch (error) { message = String(error); }
        assert(message.includes("malformed model schema") && message.includes("512"), `${field} cap was not enforced: ${message}`);
      }
      let shapeMessage = "";
      try {
        compilerModule.validateModelSchema({
          ...MODEL_SCHEMA,
          parameters: [parameter("short", {
            shape: ["s".repeat(compilerModule.MAX_MODEL_SCHEMA_TEXT_CHARACTERS + 1)],
          })],
        });
      } catch (error) { shapeMessage = String(error); }
      assert(
        shapeMessage.includes("shape dimension") && shapeMessage.includes("512"),
        `shape dimension cap was not enforced: ${shapeMessage}`,
      );
    },
  },
  {
    name: "abort during post-response hashing rejects compilation",
    fn: async () => {
      const controller = new AbortController();
      let worker;
      const compiler = client({
        workerFactory: () => {
          worker = new FakeWorker((request, active) => {
            queueMicrotask(() => {
              active.emit("message", {
                type: "compiled",
                id: request.id,
                irBytes: new TextEncoder().encode("{}").buffer,
                modelSchema: MODEL_SCHEMA,
              });
              controller.abort();
            });
          });
          queueMicrotask(() => worker.emit("message", { type: "ready", protocol: 1 }));
          return worker;
        },
      });
      await rejectedAfterTermination(
        compiler.compile("source", { signal: controller.signal }),
        worker,
        "cancelled",
      );
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

      let scenarioWorker;
      const scenarioCompiler = client({ workerFactory: () => {
        scenarioWorker = new FakeWorker();
        queueMicrotask(() => scenarioWorker.emit("message", { type: "ready", protocol: 1 }));
        return scenarioWorker;
      } });
      const scenarioController = new AbortController();
      const scenarioPending = scenarioCompiler.compileScenario(
        "main", "prior", { signal: scenarioController.signal },
      );
      scenarioController.abort();
      await rejectedAfterTermination(scenarioPending, scenarioWorker, "cancelled");
    },
  },
];
