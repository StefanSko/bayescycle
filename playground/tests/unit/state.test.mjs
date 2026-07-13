import { initialState, reduce } from "/site/src/app/state.mjs";

function assert(condition, message) { if (!condition) throw new Error(message); }

export default [
  {
    name: "source edits invalidate compilation and artifacts",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "compile-started", requestId: "c1", revision: 1 });
      state = reduce(state, { type: "compile-succeeded", requestId: "c1", revision: 1, irBytes: new Uint8Array([1]), irHash: "abc" });
      state = reduce(state, { type: "run-started", requestId: "r1", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "r1", revision: 1, artifacts: [{ name: "posterior.ndjson" }] });
      state = reduce(state, { type: "source-edited", source: "two", revision: 2 });
      assert(state.compile.status === "idle", `compile is ${state.compile.status}`);
      assert(state.run.status === "idle" && state.artifacts.length === 0, "stale run survived source edit");
    },
  },
  {
    name: "document edits retain compilation but invalidate artifacts",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "compile-started", requestId: "c1", revision: 1 });
      state = reduce(state, { type: "compile-succeeded", requestId: "c1", revision: 1, irBytes: new Uint8Array([1]), irHash: "abc" });
      state = reduce(state, { type: "documents-edited", documents: { observed: "{}", design: "", truth: "" }, revision: 2 });
      assert(state.compile.status === "compiled", "document edit discarded compilation");
      assert(state.run.status === "idle" && state.artifacts.length === 0, "document edit retained artifacts");
    },
  },
  {
    name: "stale asynchronous results are ignored",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "compile-started", requestId: "c1", revision: 1 });
      state = reduce(state, { type: "source-edited", source: "two", revision: 2 });
      state = reduce(state, { type: "compile-succeeded", requestId: "c1", revision: 1, irBytes: new Uint8Array([1]), irHash: "stale" });
      assert(state.compile.status === "idle", "stale compilation was accepted");
    },
  },
];
