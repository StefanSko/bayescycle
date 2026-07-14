import { initialState, reduce } from "/site/src/app/state.mjs";

function assert(condition, message) { if (!condition) throw new Error(message); }

export default [
  {
    name: "scoped starts reject dependencies changed during preflight",
    fn: () => {
      let state = initialState();
      const staleGeneration = {
        compileRevision: null,
        inputRevision: state.generation.inputRevision,
        settingsRevision: state.generation.settingsRevision,
        sourceKind: "fixed",
        fitLineageKey: null,
      };
      state = reduce(state, { type: "generation-input-edited", revision: 1 });
      state = reduce(state, {
        type: "generation-started", requestId: "stale-g", dependencyKey: "gk",
        guard: staleGeneration,
      });
      assert(state.generation.attempt.status === "idle", "stale generation start was accepted");

      const staleConditioning = {
        compileRevision: null,
        settingsRevision: state.conditioning.settingsRevision,
        datasetSource: "observed",
        observed: state.documents.observed,
        selectionRevision: state.generation.selectionRevision,
      };
      state = reduce(state, {
        type: "observed-input-edited",
        documents: { ...state.documents, observed: "changed" },
        revision: 2,
      });
      state = reduce(state, {
        type: "conditioning-started", requestId: "stale-f", dependencyKey: "fk",
        datasetSource: "observed", guard: staleConditioning,
      });
      assert(state.conditioning.attempt.status === "idle", "stale conditioning start was accepted");
    },
  },
  {
    name: "scoped attempts preserve successful ancestors on failure",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "generation-started", requestId: "g1", dependencyKey: "key-1" });
      state = reduce(state, {
        type: "generation-succeeded", requestId: "g1", dependencyKey: "key-1",
        collection: { sourceKind: "fixed", artifact: { name: "generated_datasets.ndjson" } },
      });
      const collection = state.generation.collection;
      state = reduce(state, { type: "generation-started", requestId: "g2", dependencyKey: "key-2" });
      state = reduce(state, { type: "generation-failed", requestId: "g2", dependencyKey: "key-2", error: "unsupported" });
      assert(state.generation.attempt.status === "failed", "failed generation attempt disappeared");
      assert(state.generation.collection === collection, "failed replacement erased the collection");

      state = reduce(state, { type: "conditioning-started", requestId: "f1", dependencyKey: "fit-1", datasetSource: "observed" });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "f1", dependencyKey: "fit-1",
        fit: { datasetSource: "observed", lineageKey: "fit-1", artifacts: [{ name: "posterior.ndjson" }] },
      });
      const fit = state.conditioning.fit;
      state = reduce(state, { type: "conditioning-started", requestId: "f2", dependencyKey: "fit-2", datasetSource: "observed" });
      state = reduce(state, { type: "conditioning-failed", requestId: "f2", dependencyKey: "fit-2", error: "bad fit" });
      assert(state.conditioning.attempt.status === "failed", "failed conditioning attempt disappeared");
      assert(state.conditioning.fit === fit, "failed replacement erased the fit");
    },
  },
  {
    name: "selection invalidates only generated conditioning descendants",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "generation-started", requestId: "g", dependencyKey: "gk" });
      state = reduce(state, {
        type: "generation-succeeded", requestId: "g", dependencyKey: "gk",
        collection: { sourceKind: "fixed", artifact: { name: "generated_datasets.ndjson" } },
      });
      const parameters = new Uint8Array([1]);
      const dataset = new Uint8Array([2]);
      state = reduce(state, { type: "selection-edited", revision: 1, index: 0, parametersBytes: parameters, datasetBytes: dataset });
      parameters[0] = 9;
      dataset[0] = 9;
      assert(state.generation.selected.parametersBytes[0] === 1, "selected parameters retained an alias");
      assert(state.generation.selected.datasetBytes[0] === 2, "selected dataset retained an alias");
      state = reduce(state, { type: "conditioning-started", requestId: "f", dependencyKey: "fk", datasetSource: "generated" });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "f", dependencyKey: "fk",
        fit: { datasetSource: "generated", lineageKey: "fk", artifacts: [{ name: "posterior.ndjson" }] },
      });
      state = reduce(state, { type: "selection-edited", revision: 2, index: 0, parametersBytes: new Uint8Array([3]), datasetBytes: new Uint8Array([4]) });
      assert(state.conditioning.fit === null, "selection edit retained a generated-data fit");
      assert(state.generation.collection !== null, "selection edit erased its collection");
    },
  },
  {
    name: "observed edits cancel observed fit and posterior generation attempts",
    fn: () => {
      let state = initialState();
      state = reduce(state, {
        type: "conditioning-started", requestId: "old", dependencyKey: "old-fit",
        datasetSource: "observed",
      });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "old", dependencyKey: "old-fit",
        fit: { datasetSource: "observed", lineageKey: "old-fit", artifacts: [] },
      });
      state = reduce(state, {
        type: "generation-started", requestId: "posterior-g", dependencyKey: "pg",
        guard: {
          compileRevision: null,
          inputRevision: state.generation.inputRevision,
          settingsRevision: state.generation.settingsRevision,
          sourceKind: "posterior",
          fitLineageKey: "old-fit",
        },
      });
      state = reduce(state, {
        type: "observed-input-edited",
        documents: { ...state.documents, observed: "changed" },
        revision: 1,
      });
      assert(state.conditioning.fit === null, "observed fit survived observed edit");
      assert(state.generation.attempt.status === "idle", "posterior generation survived source edit");
      state = reduce(state, {
        type: "generation-succeeded", requestId: "posterior-g", dependencyKey: "pg",
        collection: { sourceKind: "posterior", sourceFitLineageKey: "old-fit" },
      });
      assert(state.generation.collection === null, "stale posterior generation completion was accepted");

      state = initialState();
      state = reduce(state, {
        type: "conditioning-started", requestId: "generated", dependencyKey: "generated-fit",
        datasetSource: "generated",
      });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "generated", dependencyKey: "generated-fit",
        fit: { datasetSource: "generated", lineageKey: "generated-fit", artifacts: [] },
      });
      const generatedFit = state.conditioning.fit;
      state = reduce(state, {
        type: "conditioning-started", requestId: "replacement", dependencyKey: "observed-fit",
        datasetSource: "observed",
      });
      state = reduce(state, {
        type: "observed-input-edited",
        documents: { ...state.documents, observed: "new observed" },
        revision: 2,
      });
      assert(state.conditioning.attempt.status === "idle", "observed replacement survived edit");
      assert(state.conditioning.fit === generatedFit, "unrelated generated fit was erased");
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "replacement", dependencyKey: "observed-fit",
        fit: { datasetSource: "observed", lineageKey: "stale", artifacts: [] },
      });
      assert(state.conditioning.fit === generatedFit, "stale observed fit replaced generated fit");
    },
  },
  {
    name: "observed edits preserve generated-data fit and recovery",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "generation-started", requestId: "g", dependencyKey: "gk" });
      state = reduce(state, {
        type: "generation-succeeded", requestId: "g", dependencyKey: "gk",
        collection: { sourceKind: "fixed", artifact: { name: "generated_datasets.ndjson" } },
      });
      state = reduce(state, {
        type: "run-started", requestId: "r", revision: 0,
        operation: "sample", datasetSource: "generated",
      });
      state = reduce(state, {
        type: "run-succeeded", requestId: "r", revision: 0,
        artifacts: [{ name: "posterior.ndjson" }, { name: "recovery_check.json" }],
      });
      state = reduce(state, {
        type: "conditioning-started", requestId: "f", dependencyKey: "fk",
        datasetSource: "generated",
      });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "f", dependencyKey: "fk",
        fit: { datasetSource: "generated", lineageKey: "fk", artifacts: state.artifacts },
      });
      const fit = state.conditioning.fit;
      state = reduce(state, {
        type: "observed-input-edited",
        documents: { observed: "changed", design: "", truth: "" },
        revision: 1,
      });
      assert(state.conditioning.fit === fit, "observed edit erased generated-data fit");
      assert(state.fitDatasetSource === "generated", "observed edit erased generated fit lineage");
      assert(
        state.artifacts.some((artifact) => artifact.name === "recovery_check.json"),
        "observed edit erased generated recovery",
      );
      assert(state.generation.collection !== null, "observed edit erased independent generation");
    },
  },
  {
    name: "inference edits preserve fit and independent generation",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "generation-started", requestId: "g", dependencyKey: "gk" });
      state = reduce(state, {
        type: "generation-succeeded", requestId: "g", dependencyKey: "gk",
        collection: { sourceKind: "model-prior", artifact: { name: "generated_datasets.ndjson" } },
      });
      state = reduce(state, { type: "conditioning-started", requestId: "f", dependencyKey: "fk", datasetSource: "observed" });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "f", dependencyKey: "fk",
        fit: { datasetSource: "observed", lineageKey: "fk", artifacts: [{ name: "posterior.ndjson" }] },
      });
      const collection = state.generation.collection;
      const fit = state.conditioning.fit;
      state = reduce(state, { type: "inference-settings-edited", revision: 7 });
      assert(state.generation.collection === collection, "inference edit erased generation");
      assert(state.conditioning.fit === fit, "inference edit erased completed fit");
      assert(state.conditioning.settingsRevision === 7, "inference revision was not recorded");
    },
  },
  {
    name: "replacement fits invalidate posterior-sourced collections",
    fn: () => {
      let state = initialState();
      state = reduce(state, { type: "generation-started", requestId: "g", dependencyKey: "gk" });
      state = reduce(state, {
        type: "generation-succeeded", requestId: "g", dependencyKey: "gk",
        collection: { sourceKind: "posterior", sourceFitLineageKey: "old-fit", artifact: { name: "generated_datasets.ndjson" } },
      });
      state = reduce(state, { type: "conditioning-started", requestId: "f", dependencyKey: "new-fit", datasetSource: "observed" });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "f", dependencyKey: "new-fit",
        fit: { datasetSource: "observed", lineageKey: "new-fit", artifacts: [{ name: "posterior.ndjson" }] },
      });
      assert(state.generation.collection === null, "posterior collection survived replacement fit");
      assert(state.conditioning.fit.lineageKey === "new-fit", "replacement fit was not installed");
    },
  },
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
    name: "settings notices follow their posterior artifacts",
    fn: () => {
      const completedFit = (datasetSource) => {
        let state = initialState();
        state = reduce(state, { type: "source-edited", source: "one", revision: 1 });
        state = reduce(state, {
          type: "run-started", requestId: "sample", revision: 1,
          operation: "sample", datasetSource,
        });
        return reduce(state, {
          type: "run-succeeded", requestId: "sample", revision: 1,
          artifacts: [
            { name: "data.json" },
            { name: "posterior.ndjson" },
            { name: "simulated_data.json" },
            { name: "prior_predictive.ndjson" },
          ],
          notice: "diagnostics unavailable",
        });
      };

      let observed = completedFit("observed");
      observed = reduce(observed, { type: "generation-settings-edited", revision: 2 });
      assert(observed.notice === "diagnostics unavailable", "generation edit discarded observed fit notice");
      assert(observed.artifacts.some((artifact) => artifact.name === "posterior.ndjson"), "generation edit discarded observed fit");

      let generated = completedFit("generated");
      generated = reduce(generated, { type: "generation-settings-edited", revision: 2 });
      assert(generated.notice === null, "generation edit retained generated fit notice");
      assert(!generated.artifacts.some((artifact) => artifact.name === "posterior.ndjson"), "generation edit retained generated fit");

      let predictive = completedFit("generated");
      predictive = reduce(predictive, { type: "predictive-draws-edited", revision: 2 });
      assert(predictive.notice === "diagnostics unavailable", "predictive edit discarded fit notice");
      assert(predictive.artifacts.some((artifact) => artifact.name === "posterior.ndjson"), "predictive edit discarded fit");

      let orphaned = completedFit("observed");
      orphaned = reduce(orphaned, { type: "settings-edited", revision: 2 });
      assert(orphaned.notice === null, "orphaning settings edit retained fit notice");

      let preserved = completedFit("observed");
      preserved = Object.freeze({
        ...preserved,
        run: Object.freeze({
          status: "running", requestId: "prior", revision: 1,
          operation: "prior-predictive", datasetSource: null,
        }),
      });
      preserved = reduce(preserved, { type: "settings-edited", revision: 2 });
      assert(preserved.run.status === "running", "settings edit did not exercise preserve-run branch");
      assert(preserved.notice === null, "preserve-run settings edit retained fit notice");
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
