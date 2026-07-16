import {
  cancellationEvents,
  RunControllers,
} from "/site/src/app/run-controllers.mjs";
import { initialState, reduce } from "/site/src/app/state.mjs";
import { WorkerEngine } from "/site/src/engine/executor.mjs";

function assert(condition, message) { if (!condition) throw new Error(message); }

class PendingWorker {
  constructor() {
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
    this.messages = [];
  }
  postMessage(message) { this.messages.push(message); }
  terminate() { this.terminated = true; }
}

export default [
  {
    name: "invalidating edit aborts a superseded run worker",
    fn: async () => {
      let state = initialState();
      state = reduce(state, {
        type: "generation-started", requestId: "generation-a", dependencyKey: "key-a",
      });
      const controllers = new RunControllers();
      const controller = controllers.begin("generation", "generation-a");
      controllers.reconcile(state);
      let worker;
      const engine = new WorkerEngine("test", "wasm", "metadata", () => {
        worker = new PendingWorker();
        return worker;
      });
      const pending = engine.execute(
        { command: "generate" },
        { signal: controller.signal },
      );

      state = reduce(state, { type: "generation-input-edited", revision: 1 });
      controllers.reconcile(state);
      let error;
      try { await pending; } catch (reason) { error = reason; }
      assert(controller.signal.aborted, "superseded run signal was not aborted");
      assert(worker.terminated, "superseded run worker survived invalidation");
      assert(error?.error === "Cancelled", `unexpected worker outcome: ${String(error)}`);
    },
  },
  {
    name: "stale completion after abort cannot mutate state",
    fn: () => {
      let state = initialState();
      state = reduce(state, {
        type: "generation-started", requestId: "generation-a", dependencyKey: "key-a",
      });
      const controllers = new RunControllers();
      const controller = controllers.begin("generation", "generation-a");
      controllers.reconcile(state);
      state = reduce(state, { type: "generation-settings-scoped-edited", revision: 1 });
      controllers.reconcile(state);
      assert(controller.signal.aborted, "invalidated generation was not physically cancelled");

      state = reduce(state, {
        type: "generation-succeeded", requestId: "generation-a", dependencyKey: "key-a",
        collection: { sourceKind: "fixed", stale: true },
        selection: {
          revision: 2, index: 0,
          parametersBytes: new Uint8Array([1]), datasetBytes: new Uint8Array([2]),
        },
      });
      assert(state.generation.collection === null, "stale collection mutated state");
      assert(state.generation.selected === null, "stale selection mutated state");
    },
  },
  {
    name: "changed recompile lineage aborts and invalidates prior-model generation",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "same", revision: 1 });
      state = reduce(state, {
        type: "compile-started", requestId: "compile-old", revision: 1,
      });
      state = reduce(state, {
        type: "compile-succeeded", requestId: "compile-old", revision: 1,
        irBytes: new Uint8Array([1]), irHash: "old", modelSchema: {},
      });
      state = reduce(state, {
        type: "generation-started", requestId: "generation-old", dependencyKey: "old-key",
      });
      const controllers = new RunControllers();
      const controller = controllers.begin("generation", "generation-old");
      controllers.reconcile(state);

      // This is the existing source invalidation transition used when a
      // same-source recompile returns a different trusted hash.
      state = reduce(state, { type: "source-edited", source: "same", revision: 2 });
      controllers.reconcile(state);
      assert(controller.signal.aborted, "prior-model generation was not aborted");
      state = reduce(state, {
        type: "generation-succeeded", requestId: "generation-old", dependencyKey: "old-key",
        collection: { sourceKind: "fixed", model: "old" },
      });
      assert(state.generation.collection === null, "prior-model generation survived recompile");
    },
  },
  {
    name: "one cancellation ends every concurrently active attempt",
    fn: () => {
      let state = initialState();
      state = reduce(state, {
        type: "compile-started", requestId: "compile", revision: 0,
      });
      state = reduce(state, {
        type: "generation-started", requestId: "generation", dependencyKey: "g-key",
      });
      state = reduce(state, {
        type: "conditioning-started", requestId: "conditioning",
        dependencyKey: "c-key", datasetSource: "observed",
      });
      state = reduce(state, {
        type: "run-started", requestId: "conditioning", revision: 0,
        operation: "condition", datasetSource: "observed",
      });
      const controllers = new RunControllers();
      const compile = controllers.begin("compile", "compile");
      const generation = controllers.begin("generation", "generation");
      const conditioning = controllers.begin("conditioning", "conditioning");
      controllers.reconcile(state);

      const events = cancellationEvents(state, "Cancelled once");
      assert(events.map((event) => event.type).join(",") ===
        "run-failed,conditioning-failed,generation-failed,compile-failed",
      `unexpected cancellation events: ${events.map((event) => event.type)}`);
      for (const event of events) {
        state = reduce(state, event);
        controllers.reconcile(state);
      }
      assert(state.compile.status === "failed", "compile attempt survived cancellation");
      assert(state.generation.attempt.status === "failed", "generation attempt survived cancellation");
      assert(state.conditioning.attempt.status === "failed", "conditioning attempt survived cancellation");
      assert(state.run.status !== "running", "logical run survived cancellation");
      assert(compile.signal.aborted, "compile signal survived cancellation");
      assert(generation.signal.aborted, "generation signal survived cancellation");
      assert(conditioning.signal.aborted, "conditioning signal survived cancellation");
    },
  },
  {
    name: "stale run-start rejection is followed by conditioning cleanup",
    fn: () => {
      let state = initialState();
      state = reduce(state, {
        type: "generation-started", requestId: "generation", dependencyKey: "g-key",
      });
      state = reduce(state, {
        type: "generation-succeeded", requestId: "generation", dependencyKey: "g-key",
        collection: { sourceKind: "fixed" },
        selection: {
          revision: 0, index: 0,
          parametersBytes: new Uint8Array([1]), datasetBytes: new Uint8Array([2]),
        },
      });
      state = reduce(state, {
        type: "run-started", requestId: "old-fit", revision: 0,
        operation: "condition", datasetSource: "generated",
      });
      state = reduce(state, {
        type: "run-succeeded", requestId: "old-fit", revision: 0, artifacts: [],
      });
      const capturedProjectRevision = state.projectRevision;
      state = reduce(state, {
        type: "selection-edited", revision: 1, index: 0,
        parametersBytes: new Uint8Array([3]), datasetBytes: new Uint8Array([4]),
      });
      state = reduce(state, {
        type: "conditioning-started", requestId: "stale-start", dependencyKey: "c-key",
        datasetSource: "observed",
        guard: {
          compileRevision: null,
          settingsRevision: state.conditioning.settingsRevision,
          datasetSource: "observed",
          datasetSourceRevision: state.conditioning.datasetSourceRevision,
          observed: state.documents.observed,
          selectionRevision: 0,
        },
      });
      assert(state.conditioning.attempt.status === "running", "conditioning start was not accepted");
      state = reduce(state, {
        type: "run-started", requestId: "stale-start",
        revision: capturedProjectRevision,
        operation: "condition", datasetSource: "observed",
      });
      assert(state.run.status !== "running", "stale run start was accepted");
      state = reduce(state, {
        type: "conditioning-failed", requestId: "stale-start",
        dependencyKey: "c-key", error: "Conditioning request became stale before launch",
      });
      assert(state.conditioning.attempt.status === "failed", "nonexistent run retained an attempt");
      assert(state.run.status !== "running", "cleanup invented a logical run");
    },
  },
  {
    name: "source edit aborts its in-flight compile controller",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      const controllers = new RunControllers();
      const controller = controllers.begin("compile", "compile-a");
      state = reduce(state, {
        type: "compile-started", requestId: "compile-a", revision: 1,
      });
      controllers.reconcile(state);
      assert(!controller.signal.aborted, "current compile was cancelled");
      state = reduce(state, { type: "source-edited", source: "two", revision: 2 });
      controllers.reconcile(state);
      assert(controller.signal.aborted, "source edit retained its compile controller");
    },
  },
];
