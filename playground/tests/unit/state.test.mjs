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

      state = initialState();
      const sourceGuard = {
        compileRevision: null,
        settingsRevision: state.conditioning.settingsRevision,
        datasetSource: "observed",
        datasetSourceRevision: state.conditioning.datasetSourceRevision,
        observed: state.documents.observed,
        selectionRevision: state.generation.selectionRevision,
      };
      state = reduce(state, {
        type: "dataset-source-edited", source: "generated", revision: 3,
      });
      state = reduce(state, {
        type: "dataset-source-edited", source: "observed", revision: 4,
      });
      state = reduce(state, {
        type: "conditioning-started", requestId: "stale-source", dependencyKey: "fk",
        datasetSource: "observed", guard: sourceGuard,
      });
      assert(
        state.conditioning.attempt.status === "idle",
        "dataset-source change-and-back did not cancel stale conditioning preflight",
      );

      state = initialState();
      state = reduce(state, {
        type: "conditioning-started", requestId: "active", dependencyKey: "active-key",
        datasetSource: "observed",
        guard: {
          compileRevision: null,
          settingsRevision: state.conditioning.settingsRevision,
          datasetSource: "observed",
          datasetSourceRevision: state.conditioning.datasetSourceRevision,
          observed: state.documents.observed,
          selectionRevision: state.generation.selectionRevision,
        },
      });
      state = reduce(state, {
        type: "run-started", requestId: "active", revision: 0,
        operation: "condition", datasetSource: "observed",
      });
      state = reduce(state, {
        type: "dataset-source-edited", source: "generated", revision: 5,
      });
      assert(state.conditioning.attempt.status === "idle", "active old-source fit was retained");
      state = reduce(state, {
        type: "conditioning-committed", requestId: "active", dependencyKey: "active-key",
        revision: 0,
        fit: { datasetSource: "observed", lineageKey: "stale-source", artifacts: [] },
        artifacts: [{ name: "posterior.ndjson" }], notice: null,
      });
      assert(state.conditioning.fit === null, "old-source fit completion was accepted");
    },
  },
  {
    name: "conditioning commit installs fit and artifacts atomically",
    fn: () => {
      let state = initialState();
      state = reduce(state, {
        type: "conditioning-started", requestId: "c", dependencyKey: "ck",
        datasetSource: "observed",
      });
      state = reduce(state, {
        type: "run-started", requestId: "c", revision: 0,
        operation: "condition", datasetSource: "observed",
      });
      state = reduce(state, {
        type: "conditioning-committed", requestId: "c", dependencyKey: "ck", revision: 0,
        fit: { datasetSource: "observed", lineageKey: "lineage", artifacts: [] },
        artifacts: [{ name: "posterior.ndjson" }], notice: null,
      });
      assert(state.conditioning.fit?.lineageKey === "lineage", "atomic commit lost fit");
      assert(
        state.artifacts.some((artifact) => artifact.name === "posterior.ndjson"),
        "atomic commit lost posterior artifact",
      );

      state = initialState();
      state = reduce(state, {
        type: "conditioning-started", requestId: "stale", dependencyKey: "sk",
        datasetSource: "observed",
      });
      state = reduce(state, {
        type: "run-started", requestId: "stale", revision: 0,
        operation: "condition", datasetSource: "observed",
      });
      state = reduce(state, { type: "inference-settings-edited", revision: 1 });
      state = reduce(state, {
        type: "conditioning-committed", requestId: "stale", dependencyKey: "sk", revision: 0,
        fit: { datasetSource: "observed", lineageKey: "stale", artifacts: [] },
        artifacts: [{ name: "posterior.ndjson" }], notice: null,
      });
      assert(state.conditioning.fit === null, "stale atomic commit installed fit");
      assert(state.artifacts.length === 0, "stale atomic commit installed artifacts");
    },
  },
  {
    name: "generation success installs collection selection atomically",
    fn: () => {
      let state = initialState();
      state = reduce(state, {
        type: "generation-started", requestId: "atomic-g", dependencyKey: "atomic-key",
      });
      state = reduce(state, {
        type: "generation-succeeded", requestId: "atomic-g", dependencyKey: "atomic-key",
        collection: { sourceKind: "fixed" },
        selection: {
          revision: 1,
          index: 0,
          parametersBytes: new Uint8Array([1]),
          datasetBytes: new Uint8Array([2]),
        },
      });
      assert(state.generation.collection?.sourceKind === "fixed", "collection was not installed");
      assert(state.generation.selected?.datasetBytes[0] === 2, "selection was not installed atomically");
      state = reduce(state, {
        type: "dataset-source-edited", source: "generated", revision: 1,
      });
      state = reduce(state, {
        type: "run-started", requestId: "fit-a", revision: 0,
        operation: "condition", datasetSource: "generated",
      });
      state = reduce(state, {
        type: "run-succeeded", requestId: "fit-a", revision: 0,
        artifacts: [{ name: "posterior.ndjson" }, { name: "recovery_check.json" }],
      });
      state = reduce(state, {
        type: "conditioning-started", requestId: "fit-a", dependencyKey: "fit-a-key",
        datasetSource: "generated",
      });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "fit-a", dependencyKey: "fit-a-key",
        fit: { datasetSource: "generated", lineageKey: "fit-a", artifacts: state.artifacts },
      });
      state = reduce(state, {
        type: "generation-started", requestId: "replacement-g", dependencyKey: "replacement-key",
      });
      state = reduce(state, {
        type: "generation-succeeded", requestId: "replacement-g", dependencyKey: "replacement-key",
        collection: { sourceKind: "fixed", id: "B" },
        selection: {
          revision: 2,
          index: 0,
          parametersBytes: new Uint8Array([3]),
          datasetBytes: new Uint8Array([4]),
        },
      });
      assert(state.conditioning.fit === null, "replacement generation retained generated fit");
      assert(state.fitDatasetSource === null, "replacement generation retained fit lineage");
      assert(
        !state.artifacts.some((artifact) => artifact.name === "recovery_check.json"),
        "replacement generation retained recovery artifact",
      );
      assert(state.generation.selectionRevision === 2, "atomic selection revision did not advance");
      state = reduce(state, {
        type: "conditioning-started", requestId: "stale-a", dependencyKey: "stale-key",
        datasetSource: "generated",
        guard: {
          compileRevision: null,
          settingsRevision: state.conditioning.settingsRevision,
          datasetSource: state.conditioning.datasetSource,
          datasetSourceRevision: state.conditioning.datasetSourceRevision,
          observed: state.documents.observed,
          selectionRevision: 1,
        },
      });
      assert(state.conditioning.attempt.status === "idle", "old selection preflight was accepted");
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
      const originalCollection = state.generation.collection;
      state = reduce(state, {
        type: "generation-started", requestId: "posterior", dependencyKey: "posterior-key",
        guard: {
          compileRevision: null,
          inputRevision: state.generation.inputRevision,
          settingsRevision: state.generation.settingsRevision,
          sourceKind: "posterior",
          fitLineageKey: "fk",
        },
      });
      state = reduce(state, { type: "selection-edited", revision: 2, index: 0, parametersBytes: new Uint8Array([3]), datasetBytes: new Uint8Array([4]) });
      assert(state.conditioning.fit === null, "selection edit retained a generated-data fit");
      assert(state.generation.collection === originalCollection, "selection edit erased its collection");
      assert(state.generation.attempt.status === "idle", "selection edit retained posterior generation");
      state = reduce(state, {
        type: "generation-succeeded", requestId: "posterior", dependencyKey: "posterior-key",
        collection: { sourceKind: "posterior", sourceFitLineageKey: "fk" },
      });
      assert(
        state.generation.collection === originalCollection,
        "stale posterior generation replaced selected collection",
      );
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
      state = reduce(state, {
        type: "conditioning-started", requestId: "old", dependencyKey: "old-fit",
        datasetSource: "observed",
      });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "old", dependencyKey: "old-fit",
        fit: { datasetSource: "observed", lineageKey: "old-fit", artifacts: [] },
      });
      state = reduce(state, { type: "generation-started", requestId: "g", dependencyKey: "gk" });
      state = reduce(state, {
        type: "generation-succeeded", requestId: "g", dependencyKey: "gk",
        collection: { sourceKind: "posterior", sourceFitLineageKey: "old-fit", artifact: { name: "generated_datasets.ndjson" } },
      });
      state = reduce(state, {
        type: "generation-started", requestId: "pending-old", dependencyKey: "pending-key",
        guard: {
          compileRevision: null,
          inputRevision: state.generation.inputRevision,
          settingsRevision: state.generation.settingsRevision,
          sourceKind: "posterior",
          fitLineageKey: "old-fit",
        },
      });
      state = reduce(state, { type: "conditioning-started", requestId: "f", dependencyKey: "new-fit", datasetSource: "observed" });
      state = reduce(state, {
        type: "conditioning-succeeded", requestId: "f", dependencyKey: "new-fit",
        fit: { datasetSource: "observed", lineageKey: "new-fit", artifacts: [{ name: "posterior.ndjson" }] },
      });
      assert(state.generation.collection === null, "posterior collection survived replacement fit");
      assert(state.generation.attempt.status === "idle", "old-fit generation attempt survived");
      assert(state.conditioning.fit.lineageKey === "new-fit", "replacement fit was not installed");
      state = reduce(state, {
        type: "generation-succeeded", requestId: "pending-old", dependencyKey: "pending-key",
        collection: { sourceKind: "posterior", sourceFitLineageKey: "old-fit" },
      });
      assert(state.generation.collection === null, "old-fit generation completion was accepted");
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
      const modelSchema = {
        schema_format: "bayescycle.playground.model-schema.v0",
        parameters: [], data: [], observed: [],
      };
      state = reduce(state, {
        type: "compile-succeeded", requestId: "c1", revision: 1,
        irBytes: new Uint8Array([1]), irHash: "abc", modelSchema,
      });
      state = reduce(state, { type: "documents-edited", documents: { observed: "{}", design: "", truth: "" }, revision: 2 });
      assert(state.compile.status === "compiled", "document edit discarded compilation");
      assert(state.compile.modelSchema === modelSchema, "compile state did not retain model schema");
      assert(state.run.status === "idle" && state.artifacts.length === 0, "document edit retained artifacts");
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
