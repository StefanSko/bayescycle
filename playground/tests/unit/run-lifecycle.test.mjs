import { RunControllers } from "/site/src/app/run-controllers.mjs";
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
