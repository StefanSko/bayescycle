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
    name: "sampler edits invalidate runs but retain compilation",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "compile-started", requestId: "c1", revision: 1 });
      state = reduce(state, { type: "compile-succeeded", requestId: "c1", revision: 1, irBytes: new Uint8Array([1]), irHash: "abc" });
      state = reduce(state, { type: "run-started", requestId: "r1", revision: 1 });
      state = reduce(state, { type: "settings-edited", revision: 2 });
      state = reduce(state, { type: "run-succeeded", requestId: "r1", revision: 1, artifacts: [{ name: "posterior.ndjson" }] });
      assert(state.compile.status === "compiled", "settings edit discarded compilation");
      assert(state.run.status === "idle" && state.artifacts.length === 0, "stale run survived settings edit");
    },
  },
  {
    name: "later runs clear stale follow-up notices",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "run-started", requestId: "r1", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "r1", revision: 1, artifacts: [], notice: "diagnostics unavailable" });
      assert(state.notice === "diagnostics unavailable", "follow-up notice was not retained");
      state = reduce(state, { type: "run-started", requestId: "r2", revision: 1 });
      assert(state.notice === null, "run start retained stale notice");
      state = reduce(state, { type: "run-succeeded", requestId: "r2", revision: 1, artifacts: [] });
      assert(state.notice === null, "successful run restored stale notice");
    },
  },
  {
    name: "successful follow-ups preserve earlier artifacts",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "run-started", requestId: "simulate", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "simulate", revision: 1, artifacts: [{ name: "simulated_data.json" }] });
      state = reduce(state, { type: "run-started", requestId: "sample", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "sample", revision: 1, artifacts: [{ name: "posterior.ndjson" }] });
      assert(state.artifacts.map((artifact) => artifact.name).join(",") === "simulated_data.json,posterior.ndjson", "follow-up erased an earlier artifact");
    },
  },
  {
    name: "a replacement posterior drops artifacts derived from the old fit",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "run-started", requestId: "old", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "old", revision: 1, artifacts: [
        { name: "simulated_data.json" }, { name: "posterior.ndjson", generation: "old" },
        { name: "diagnostics.json" }, { name: "recovery_check.json" },
        { name: "posterior_predictive.ndjson" },
      ] });
      state = reduce(state, { type: "run-started", requestId: "new", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "new", revision: 1, artifacts: [
        { name: "model.ir.json" }, { name: "data.json" },
        { name: "posterior.ndjson", generation: "new" },
      ] });
      const names = state.artifacts.map((artifact) => artifact.name);
      assert(names.includes("simulated_data.json"), "new posterior erased independent simulated data");
      assert(!names.includes("diagnostics.json") && !names.includes("recovery_check.json") &&
        !names.includes("posterior_predictive.ndjson"), `old fit artifacts survived: ${names}`);
      assert(state.artifacts.find((artifact) => artifact.name === "posterior.ndjson").generation === "new", "old posterior survived");
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
