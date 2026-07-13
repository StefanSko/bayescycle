export function initialState() {
  return Object.freeze({
    source: "",
    documents: Object.freeze({ observed: "", design: "", truth: "" }),
    sourceRevision: 0,
    projectRevision: 0,
    compile: Object.freeze({ status: "idle" }),
    run: Object.freeze({ status: "idle" }),
    artifacts: Object.freeze([]),
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
      });
    case "documents-edited":
      return freeze({
        ...state,
        documents: event.documents,
        projectRevision: event.revision,
        run: { status: "idle" },
        artifacts: [],
      });
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
      return freeze({ ...state, run: { status: "running", requestId: event.requestId, revision: event.revision } });
    case "run-succeeded":
      if (!matches(state.run, event, "running") || event.revision !== state.projectRevision) return state;
      return freeze({ ...state, run: { status: "completed", revision: event.revision }, artifacts: mergeArtifacts(state.artifacts, event.artifacts) });
    case "run-failed":
      if (!matches(state.run, event, "running")) return state;
      return freeze({ ...state, run: { status: "failed", error: event.error } });
    default:
      return state;
  }
}

function mergeArtifacts(existing, added) {
  const replaced = new Set(added.map((artifact) => artifact.name));
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
