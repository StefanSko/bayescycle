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

export function initialState() {
  return Object.freeze({
    source: "",
    documents: Object.freeze({ observed: "", design: "", truth: "" }),
    sourceRevision: 0,
    projectRevision: 0,
    compile: Object.freeze({ status: "idle" }),
    run: Object.freeze({ status: "idle" }),
    artifacts: Object.freeze([]),
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
        notice: null,
      });
    case "documents-edited":
      return freeze({
        ...state,
        documents: event.documents,
        projectRevision: event.revision,
        run: { status: "idle" },
        artifacts: [],
        notice: null,
      });
    case "settings-edited":
      return invalidateSettings(
        state,
        event,
        FIT_DESCENDANT_ARTIFACTS,
        state.run.status === "running" &&
          ["simulate", "prior-predictive"].includes(state.run.operation),
      );
    case "generation-settings-edited":
      return invalidateSettings(
        state,
        event,
        GENERATION_DESCENDANT_ARTIFACTS,
        state.run.status === "running" && state.run.operation === "sample",
      );
    case "compile-started":
      if (event.revision !== state.sourceRevision) return state;
      return freeze({ ...state, compile: { status: "compiling", requestId: event.requestId, revision: event.revision }, run: { status: "idle" }, artifacts: [] });
    case "compile-succeeded":
      if (!matches(state.compile, event, "compiling") || event.revision !== state.sourceRevision) return state;
      return freeze({ ...state, compile: { status: "compiled", irBytes: event.irBytes, irHash: event.irHash, revision: event.revision }, run: { status: "idle" }, artifacts: [] });
    case "compile-failed":
      if (!matches(state.compile, event, "compiling")) return state;
      return freeze({ ...state, compile: { status: "failed", error: event.error }, run: { status: "idle" }, artifacts: [] });
    case "run-started":
      if (event.revision !== state.projectRevision) return state;
      return freeze({ ...state, run: { status: "running", requestId: event.requestId, revision: event.revision, operation: event.operation }, notice: null });
    case "run-succeeded":
      if (!matches(state.run, event, "running") || event.revision !== state.projectRevision) return state;
      return freeze({ ...state, run: { status: "completed", revision: event.revision }, artifacts: mergeArtifacts(state.artifacts, event.artifacts), notice: event.notice ?? null });
    case "run-failed":
      if (!matches(state.run, event, "running")) return state;
      return freeze({ ...state, run: { status: "failed", error: event.error }, notice: null });
    default:
      return state;
  }
}

function invalidateSettings(state, event, invalidatedArtifacts, preserveRun) {
  return freeze({
    ...state,
    ...(preserveRun ? {} : {
      projectRevision: event.revision,
      run: { status: "idle" },
      notice: null,
    }),
    artifacts: state.artifacts.filter(
      (artifact) => !invalidatedArtifacts.includes(artifact.name),
    ),
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
