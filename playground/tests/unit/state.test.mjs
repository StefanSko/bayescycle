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
      state = reduce(state, { type: "run-started", requestId: "complete", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "complete", revision: 1, artifacts: [
        { name: "model.ir.json" },
        { name: "data.json" },
        { name: "posterior.ndjson" },
        { name: "diagnostics.json" },
        { name: "recovery_check.json" },
        { name: "prior_predictive.ndjson" },
        { name: "simulated_data.json" },
        { name: "posterior_predictive.ndjson" },
      ] });
      state = reduce(state, { type: "run-started", requestId: "stale", revision: 1, operation: "sample" });
      state = reduce(state, { type: "settings-edited", revision: 2 });
      assert(state.compile.status === "compiled", "settings edit discarded compilation");
      assert(state.projectRevision === 2, `project revision is ${state.projectRevision}`);
      assert(state.run.status === "idle" && state.notice === null, "settings edit retained run state");
      const names = state.artifacts.map((artifact) => artifact.name).join(",");
      assert(names === "model.ir.json,prior_predictive.ndjson,simulated_data.json", `unexpected retained artifacts: ${names}`);
      const edited = state;
      state = reduce(state, { type: "run-succeeded", requestId: "stale", revision: 1, artifacts: [{ name: "stale.json" }] });
      assert(state === edited, "stale completion at the old revision was accepted");
    },
  },
  {
    name: "generation settings selectively invalidate artifacts and stale runs",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, {
        type: "run-started", requestId: "complete", revision: 1,
        operation: "sample", datasetSource: "observed",
      });
      state = reduce(state, { type: "run-succeeded", requestId: "complete", revision: 1, artifacts: [
        { name: "model.ir.json" },
        { name: "data.json" },
        { name: "posterior.ndjson" },
        { name: "diagnostics.json" },
        { name: "recovery_check.json" },
        { name: "prior_predictive.ndjson" },
        { name: "simulated_data.json" },
        { name: "posterior_predictive.ndjson" },
      ] });
      assert(state.fitDatasetSource === "observed", "observed fit lineage was not recorded");
      state = reduce(state, { type: "run-started", requestId: "stale", revision: 1, operation: "simulate" });
      state = reduce(state, { type: "generation-settings-edited", revision: 2 });
      assert(state.projectRevision === 2, `project revision is ${state.projectRevision}`);
      assert(state.run.status === "idle" && state.notice === null, "generation edit retained run state");
      assert(state.fitDatasetSource === "observed", "generation edit discarded observed fit lineage");
      const names = state.artifacts.map((artifact) => artifact.name).join(",");
      assert(names === "model.ir.json,data.json,posterior.ndjson,diagnostics.json,recovery_check.json", `unexpected retained artifacts: ${names}`);
      const edited = state;
      state = reduce(state, { type: "run-succeeded", requestId: "stale", revision: 1, artifacts: [{ name: "stale.json" }] });
      assert(state === edited, "stale completion at the old revision was accepted");
    },
  },
  {
    name: "generation edits invalidate a completed generated-data fit",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, {
        type: "run-started", requestId: "sample", revision: 1,
        operation: "sample", datasetSource: "generated",
      });
      state = reduce(state, { type: "run-succeeded", requestId: "sample", revision: 1, artifacts: [
        { name: "model.ir.json" },
        { name: "data.json" },
        { name: "posterior.ndjson" },
        { name: "diagnostics.json" },
        { name: "recovery_check.json" },
        { name: "prior_predictive.ndjson" },
        { name: "simulated_data.json" },
        { name: "posterior_predictive.ndjson" },
      ] });
      assert(state.fitDatasetSource === "generated", "generated fit lineage was not recorded");
      state = reduce(state, { type: "generation-settings-edited", revision: 2 });
      assert(state.projectRevision === 2 && state.run.status === "idle", "generation edit did not invalidate completed fit");
      assert(state.fitDatasetSource === null, "generated fit lineage survived invalidation");
      assert(state.artifacts.map((artifact) => artifact.name).join(",") === "model.ir.json", "generated fit descendants survived invalidation");
    },
  },
  {
    name: "generation edits orphan a running generated-data fit",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "run-started", requestId: "setup", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "setup", revision: 1, artifacts: [
        { name: "simulated_data.json" },
      ] });
      state = reduce(state, {
        type: "run-started", requestId: "sample", revision: 1,
        operation: "sample", datasetSource: "generated",
      });
      state = reduce(state, { type: "generation-settings-edited", revision: 2 });
      assert(state.projectRevision === 2 && state.run.status === "idle", "generation edit retained generated fit");
      assert(state.artifacts.length === 0, "stale generated data survived edit");
      const edited = state;
      state = reduce(state, { type: "run-succeeded", requestId: "sample", revision: 1, artifacts: [
        { name: "posterior.ndjson" },
      ] });
      assert(state === edited, "stale generated fit completion was accepted");
    },
  },
  {
    name: "generation edits preserve an independent running sample",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "run-started", requestId: "setup", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "setup", revision: 1, artifacts: [
        { name: "data.json" },
        { name: "prior_predictive.ndjson" },
        { name: "simulated_data.json" },
        { name: "posterior_predictive.ndjson" },
      ] });
      state = reduce(state, {
        type: "run-started", requestId: "sample", revision: 1,
        operation: "sample", datasetSource: "observed",
      });
      state = reduce(state, { type: "generation-settings-edited", revision: 2 });
      assert(state.projectRevision === 1, `project revision changed to ${state.projectRevision}`);
      assert(state.run.status === "running" && state.run.requestId === "sample", "generation edit orphaned sample");
      assert(state.notice === null, "generation edit changed the active run notice");
      assert(state.artifacts.map((artifact) => artifact.name).join(",") === "data.json", "generation descendants survived edit");
      state = reduce(state, { type: "run-succeeded", requestId: "sample", revision: 1, artifacts: [
        { name: "model.ir.json" }, { name: "data.json" }, { name: "posterior.ndjson" },
      ] });
      assert(state.run.status === "completed", "sample completion was rejected");
      assert(state.fitDatasetSource === "observed", "observed fit lineage was not recorded");
      assert(state.artifacts.some((artifact) => artifact.name === "posterior.ndjson"), "sample artifacts were not merged");
    },
  },
  {
    name: "sampler edits preserve an independent prior predictive run",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, { type: "run-started", requestId: "setup", revision: 1 });
      state = reduce(state, { type: "run-succeeded", requestId: "setup", revision: 1, artifacts: [
        { name: "model.ir.json" },
        { name: "data.json" },
        { name: "posterior.ndjson" },
        { name: "diagnostics.json" },
        { name: "recovery_check.json" },
        { name: "prior_predictive.ndjson", generation: "old" },
        { name: "simulated_data.json" },
        { name: "posterior_predictive.ndjson" },
      ] });
      state = reduce(state, { type: "run-started", requestId: "prior", revision: 1, operation: "prior-predictive" });
      state = reduce(state, { type: "settings-edited", revision: 2 });
      assert(state.projectRevision === 1, `project revision changed to ${state.projectRevision}`);
      assert(state.run.status === "running" && state.run.requestId === "prior", "sampler edit orphaned prior run");
      assert(state.notice === null, "sampler edit changed the active run notice");
      const names = state.artifacts.map((artifact) => artifact.name).join(",");
      assert(names === "model.ir.json,prior_predictive.ndjson,simulated_data.json", `fit descendants survived: ${names}`);
      state = reduce(state, { type: "run-succeeded", requestId: "prior", revision: 1, artifacts: [
        { name: "prior_predictive.ndjson", generation: "new" },
      ] });
      assert(state.run.status === "completed", "prior completion was rejected");
      assert(state.artifacts.find((artifact) => artifact.name === "prior_predictive.ndjson").generation === "new", "prior artifacts were not merged");
    },
  },
  {
    name: "predictive draw edits preserve fits and running samples",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, {
        type: "run-started", requestId: "setup", revision: 1,
        operation: "sample", datasetSource: "generated",
      });
      state = reduce(state, { type: "run-succeeded", requestId: "setup", revision: 1, artifacts: [
        { name: "model.ir.json" },
        { name: "data.json" },
        { name: "posterior.ndjson" },
        { name: "diagnostics.json" },
        { name: "recovery_check.json" },
        { name: "prior_predictive.ndjson" },
        { name: "simulated_data.json" },
        { name: "posterior_predictive.ndjson" },
      ] });
      state = reduce(state, {
        type: "run-started", requestId: "sample", revision: 1,
        operation: "sample", datasetSource: "generated",
      });
      state = reduce(state, { type: "predictive-draws-edited", revision: 2 });
      assert(state.projectRevision === 1, `project revision changed to ${state.projectRevision}`);
      assert(state.run.status === "running" && state.run.requestId === "sample", "predictive edit orphaned sample");
      assert(state.fitDatasetSource === "generated", "predictive edit discarded fit lineage");
      const names = state.artifacts.map((artifact) => artifact.name);
      assert(!names.includes("prior_predictive.ndjson"), "prior predictive artifact survived edit");
      for (const retained of ["data.json", "posterior.ndjson", "simulated_data.json", "posterior_predictive.ndjson"]) {
        assert(names.includes(retained), `${retained} was incorrectly invalidated`);
      }
      state = reduce(state, { type: "run-succeeded", requestId: "sample", revision: 1, artifacts: [
        { name: "model.ir.json" }, { name: "data.json" }, { name: "posterior.ndjson" },
      ] });
      assert(state.run.status === "completed", "sample completion was rejected");
      assert(state.fitDatasetSource === "generated", "accepted sample lost fit lineage");
    },
  },
  {
    name: "predictive draw edits orphan running prior generation",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
      state = reduce(state, {
        type: "run-started", requestId: "setup", revision: 1,
        operation: "sample", datasetSource: "generated",
      });
      state = reduce(state, { type: "run-succeeded", requestId: "setup", revision: 1, artifacts: [
        { name: "data.json" },
        { name: "posterior.ndjson" },
        { name: "simulated_data.json" },
        { name: "prior_predictive.ndjson" },
      ] });
      state = reduce(state, { type: "run-started", requestId: "prior", revision: 1, operation: "prior-predictive" });
      state = reduce(state, { type: "predictive-draws-edited", revision: 2 });
      assert(state.projectRevision === 2 && state.run.status === "idle", "predictive edit retained prior run");
      assert(state.fitDatasetSource === "generated", "predictive edit discarded fit lineage");
      const names = state.artifacts.map((artifact) => artifact.name).join(",");
      assert(names === "data.json,posterior.ndjson,simulated_data.json", `unexpected retained artifacts: ${names}`);
      const edited = state;
      state = reduce(state, { type: "run-succeeded", requestId: "prior", revision: 1, artifacts: [
        { name: "prior_predictive.ndjson" },
      ] });
      assert(state === edited, "stale prior completion was accepted");
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
