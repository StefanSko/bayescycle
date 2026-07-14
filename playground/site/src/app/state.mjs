const FIT_DESCENDANT_ARTIFACTS = [
  "data.json",
  "posterior.ndjson",
  "diagnostics.json",
  "recovery_check.json",
  "posterior_predictive.ndjson",
];
const GENERATION_DESCENDANT_ARTIFACTS = [
  "prior_predictive.ndjson",
  "simulated_data.json",
  "posterior_predictive.ndjson",
];
const PRIOR_PREDICTIVE_ARTIFACTS = ["prior_predictive.ndjson"];

export function initialState() {
  return Object.freeze({
    source: "",
    documents: Object.freeze({ observed: "", design: "", truth: "" }),
    sourceRevision: 0,
    projectRevision: 0,
    compile: Object.freeze({ status: "idle" }),
    run: Object.freeze({ status: "idle" }),
    artifacts: Object.freeze([]),
    fitDatasetSource: null,
    notice: null,
  });
}

export function reduce(state, event) {
  switch (event.type) {
    case "source-edited":
      return freeze({
        ...state,
        source: event.source,
        sourceRevision: event.revision,
        projectRevision: event.revision,
        compile: { status: "idle" },
        run: { status: "idle" },
        artifacts: [],
        fitDatasetSource: null,
        notice: null,
      });
    case "documents-edited":
      return freeze({
        ...state,
        documents: event.documents,
        projectRevision: event.revision,
        run: { status: "idle" },
        artifacts: [],
        fitDatasetSource: null,
        notice: null,
      });
    case "settings-edited":
      return invalidateSettings(
        state,
        event,
        FIT_DESCENDANT_ARTIFACTS,
        state.run.status === "running" &&
          ["simulate", "prior-predictive"].includes(state.run.operation),
        null,
      );
    case "generation-settings-edited": {
      const generatedLineage = state.fitDatasetSource === "generated";
      return invalidateSettings(
        state,
        event,
        generatedLineage
          ? [...GENERATION_DESCENDANT_ARTIFACTS, ...FIT_DESCENDANT_ARTIFACTS]
          : GENERATION_DESCENDANT_ARTIFACTS,
        state.run.status === "running" && state.run.operation === "sample" &&
          state.run.datasetSource === "observed",
        generatedLineage ? null : state.fitDatasetSource,
      );
    }
    case "predictive-draws-edited":
      return invalidateSettings(
        state,
        event,
        PRIOR_PREDICTIVE_ARTIFACTS,
        state.run.status === "running" &&
          state.run.operation !== "prior-predictive",
        state.fitDatasetSource,
      );
    case "compile-started":
      if (event.revision !== state.sourceRevision) return state;
      return freeze({ ...state, compile: { status: "compiling", requestId: event.requestId, revision: event.revision }, run: { status: "idle" }, artifacts: [], fitDatasetSource: null });
    case "compile-succeeded":
      if (!matches(state.compile, event, "compiling") || event.revision !== state.sourceRevision) return state;
      return freeze({ ...state, compile: { status: "compiled", irBytes: event.irBytes, irHash: event.irHash, revision: event.revision }, run: { status: "idle" }, artifacts: [], fitDatasetSource: null });
    case "compile-failed":
      if (!matches(state.compile, event, "compiling")) return state;
      return freeze({ ...state, compile: { status: "failed", error: event.error }, run: { status: "idle" }, artifacts: [], fitDatasetSource: null });
    case "run-started":
      if (event.revision !== state.projectRevision) return state;
      return freeze({ ...state, run: { status: "running", requestId: event.requestId, revision: event.revision, operation: event.operation, datasetSource: event.datasetSource ?? null }, notice: null });
    case "run-succeeded":
      if (!matches(state.run, event, "running") || event.revision !== state.projectRevision) return state;
      return freeze({
        ...state,
        run: { status: "completed", revision: event.revision },
        artifacts: mergeArtifacts(state.artifacts, event.artifacts),
        fitDatasetSource: state.run.operation === "sample"
          ? state.run.datasetSource
          : state.fitDatasetSource,
        notice: event.notice ?? null,
      });
    case "run-failed":
      if (!matches(state.run, event, "running")) return state;
      return freeze({ ...state, run: { status: "failed", error: event.error }, notice: null });
    default:
      return state;
  }
}

function invalidateSettings(
  state,
  event,
  invalidatedArtifacts,
  preserveRun,
  fitDatasetSource,
) {
  return freeze({
    ...state,
    ...(preserveRun ? {} : {
      projectRevision: event.revision,
      run: { status: "idle" },
    }),
    artifacts: state.artifacts.filter(
      (artifact) => !invalidatedArtifacts.includes(artifact.name),
    ),
    fitDatasetSource,
    notice: invalidatedArtifacts.includes("posterior.ndjson") ? null : state.notice,
  });
}

function mergeArtifacts(existing, added) {
  const replaced = new Set(added.map((artifact) => artifact.name));
  if (replaced.has("posterior.ndjson")) {
    for (const dependent of [
      "diagnostics.json",
      "recovery_check.json",
      "posterior_predictive.ndjson",
    ]) replaced.add(dependent);
  }
  return [...existing.filter((artifact) => !replaced.has(artifact.name)), ...added];
}

function matches(active, event, status) {
  return active.status === status && active.requestId === event.requestId && active.revision === event.revision;
}

function freeze(state) {
  return Object.freeze({
    ...state,
    documents: Object.freeze({ ...state.documents }),
    compile: Object.freeze({ ...state.compile }),
    run: Object.freeze({ ...state.run }),
    artifacts: Object.freeze([...state.artifacts]),
  });
}
