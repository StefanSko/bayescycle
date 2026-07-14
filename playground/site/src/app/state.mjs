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
    generation: freezeGeneration({
      attempt: { status: "idle" }, collection: null, selected: null,
      inputRevision: 0, settingsRevision: 0, selectionRevision: 0,
    }),
    conditioning: freezeConditioning({
      attempt: { status: "idle" }, fit: null, settingsRevision: 0,
    }),
    artifacts: Object.freeze([]),
    fitDatasetSource: null,
    notice: null,
  });
}

export function reduce(state, event) {
  const scoped = reduceScopedWorkflow(state, event);
  if (scoped !== null) return freeze(scoped);
  switch (event.type) {
    case "source-edited":
      return freeze({
        ...state,
        source: event.source,
        sourceRevision: event.revision,
        projectRevision: event.revision,
        compile: { status: "idle" },
        run: { status: "idle" },
        generation: {
          attempt: { status: "idle" }, collection: null, selected: null,
          inputRevision: event.revision, settingsRevision: 0, selectionRevision: 0,
        },
        conditioning: {
          attempt: { status: "idle" }, fit: null, settingsRevision: 0,
        },
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
        generation: {
          ...state.generation,
          attempt: { status: "idle" }, collection: null, selected: null,
          inputRevision: event.revision,
        },
        conditioning: {
          ...state.conditioning,
          attempt: { status: "idle" }, fit: null,
        },
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
        state.run.status === "running" &&
          ["sample", "condition"].includes(state.run.operation) &&
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
        fitDatasetSource: ["sample", "condition"].includes(state.run.operation)
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

function reduceScopedWorkflow(state, event) {
  switch (event.type) {
    case "generation-started":
      if (event.guard !== undefined && !matchesGenerationGuard(state, event.guard)) {
        return state;
      }
      return {
        ...state,
        generation: {
          ...state.generation,
          attempt: {
            status: "running", requestId: event.requestId,
            dependencyKey: event.dependencyKey,
            sourceKind: event.guard?.sourceKind ?? null,
            sourceFitLineageKey: event.guard?.fitLineageKey ?? null,
          },
        },
      };
    case "generation-succeeded":
      if (!matchesAttempt(state.generation.attempt, event)) return state;
      return {
        ...state,
        generation: {
          ...state.generation,
          attempt: { status: "completed", dependencyKey: event.dependencyKey },
          collection: freezeCollection(event.collection),
          selected: null,
        },
        conditioning: clearGeneratedFit(state.conditioning),
      };
    case "generation-failed":
      if (!matchesAttempt(state.generation.attempt, event)) return state;
      return {
        ...state,
        generation: {
          ...state.generation,
          attempt: { status: "failed", dependencyKey: event.dependencyKey, error: event.error },
        },
      };
    case "generation-input-edited":
    case "parameter-source-edited":
    case "generation-settings-scoped-edited": {
      const settings = event.type === "generation-settings-scoped-edited";
      const generatedFit = state.fitDatasetSource === "generated" ||
        state.conditioning.fit?.datasetSource === "generated" ||
        state.conditioning.attempt.datasetSource === "generated";
      return {
        ...state,
        ...(event.documents === undefined ? {} : {
          documents: event.documents,
        }),
        ...(generatedFit ? {
          projectRevision: event.revision,
          run: { status: "idle" },
          artifacts: state.artifacts.filter(
            (artifact) => !FIT_DESCENDANT_ARTIFACTS.includes(artifact.name),
          ),
          fitDatasetSource: null,
          notice: null,
        } : {}),
        generation: {
          ...state.generation,
          attempt: { status: "idle" }, collection: null, selected: null,
          ...(settings
            ? { settingsRevision: event.revision }
            : { inputRevision: event.revision }),
        },
        conditioning: clearGeneratedFit(state.conditioning),
      };
    }
    case "observed-input-edited": {
      const runningObserved = state.conditioning.attempt.status === "running" &&
        state.conditioning.attempt.datasetSource === "observed";
      const runningGenerated = state.conditioning.attempt.status === "running" &&
        state.conditioning.attempt.datasetSource === "generated";
      const completedGenerated = state.fitDatasetSource === "generated" ||
        state.conditioning.fit?.datasetSource === "generated";
      const generatedRun = state.run.status === "running" &&
        ["sample", "condition"].includes(state.run.operation) &&
        state.run.datasetSource === "generated";
      const posteriorGeneration =
        state.generation.collection?.sourceKind === "posterior" ||
        (state.generation.attempt.status === "running" &&
          state.generation.attempt.sourceKind === "posterior");
      if (runningGenerated || generatedRun || (completedGenerated && !runningObserved)) {
        return {
          ...state,
          documents: event.documents,
        };
      }
      if (runningObserved && completedGenerated) {
        return {
          ...state,
          documents: event.documents,
          projectRevision: event.revision,
          run: { status: "idle" },
          generation: posteriorGeneration
            ? { ...state.generation, attempt: { status: "idle" }, collection: null, selected: null }
            : state.generation,
          conditioning: {
            ...state.conditioning,
            attempt: { status: "idle" },
          },
        };
      }
      return {
        ...state,
        documents: event.documents,
        projectRevision: event.revision,
        run: { status: "idle" },
        artifacts: state.artifacts.filter(
          (artifact) => !FIT_DESCENDANT_ARTIFACTS.includes(artifact.name),
        ),
        fitDatasetSource: null,
        generation: posteriorGeneration
          ? { ...state.generation, attempt: { status: "idle" }, collection: null, selected: null }
          : state.generation,
        conditioning: {
          ...state.conditioning,
          attempt: { status: "idle" }, fit: null,
        },
        notice: null,
      };
    }
    case "selection-edited": {
      if (state.generation.collection === null) return state;
      const generatedFit = state.fitDatasetSource === "generated" ||
        state.conditioning.fit?.datasetSource === "generated" ||
        state.conditioning.attempt.datasetSource === "generated" ||
        (state.run.status === "running" && state.run.datasetSource === "generated");
      return {
        ...state,
        ...(generatedFit ? {
          projectRevision: event.revision,
          run: { status: "idle" },
          artifacts: state.artifacts.filter(
            (artifact) => !FIT_DESCENDANT_ARTIFACTS.includes(artifact.name),
          ),
          fitDatasetSource: null,
          notice: null,
        } : {}),
        generation: {
          ...state.generation,
          selected: selectedValue(event),
          selectionRevision: event.revision,
        },
        conditioning: clearGeneratedFit(state.conditioning),
      };
    }
    case "conditioning-started":
      if (event.guard !== undefined && !matchesConditioningGuard(state, event.guard)) {
        return state;
      }
      return {
        ...state,
        conditioning: {
          ...state.conditioning,
          attempt: {
            status: "running", requestId: event.requestId,
            dependencyKey: event.dependencyKey, datasetSource: event.datasetSource,
          },
        },
      };
    case "conditioning-succeeded": {
      if (!matchesAttempt(state.conditioning.attempt, event)) return state;
      const fit = freezeFit(event.fit);
      const collection = state.generation.collection;
      const stalePosteriorCollection = collection?.sourceKind === "posterior" &&
        collection.sourceFitLineageKey !== fit.lineageKey;
      return {
        ...state,
        generation: stalePosteriorCollection
          ? { ...state.generation, attempt: { status: "idle" }, collection: null, selected: null }
          : state.generation,
        conditioning: {
          ...state.conditioning,
          attempt: { status: "completed", dependencyKey: event.dependencyKey },
          fit,
        },
      };
    }
    case "conditioning-failed":
      if (!matchesAttempt(state.conditioning.attempt, event)) return state;
      return {
        ...state,
        conditioning: {
          ...state.conditioning,
          attempt: { status: "failed", dependencyKey: event.dependencyKey, error: event.error },
        },
      };
    case "inference-settings-edited":
      return {
        ...state,
        ...(state.conditioning.attempt.status === "running" ? {
          projectRevision: event.revision,
          run: { status: "idle" },
        } : {}),
        conditioning: {
          ...state.conditioning,
          attempt: { status: "idle" }, settingsRevision: event.revision,
        },
      };
    default:
      return null;
  }
}

function matchesAttempt(attempt, event) {
  return attempt.status === "running" && attempt.requestId === event.requestId &&
    attempt.dependencyKey === event.dependencyKey;
}

function clearGeneratedFit(conditioning) {
  return conditioning.fit?.datasetSource === "generated" ||
    conditioning.attempt.datasetSource === "generated"
    ? { ...conditioning, attempt: { status: "idle" }, fit: null }
    : conditioning;
}

function matchesGenerationGuard(state, guard) {
  return guard.compileRevision === (state.compile.revision ?? null) &&
    guard.inputRevision === state.generation.inputRevision &&
    guard.settingsRevision === state.generation.settingsRevision &&
    (guard.sourceKind !== "posterior" ||
      guard.fitLineageKey === (state.conditioning.fit?.lineageKey ?? null));
}

function matchesConditioningGuard(state, guard) {
  if (guard.compileRevision !== (state.compile.revision ?? null) ||
      guard.settingsRevision !== state.conditioning.settingsRevision) return false;
  if (guard.datasetSource === "generated") {
    return guard.selectionRevision === state.generation.selectionRevision;
  }
  return guard.observed === state.documents.observed;
}

function freezeCollection(collection) {
  return Object.freeze({ ...collection });
}

function freezeFit(fit) {
  return Object.freeze({
    ...fit,
    artifacts: Object.freeze([...(fit.artifacts ?? [])]),
  });
}

function selectedValue(event) {
  const parameters = Uint8Array.from(event.parametersBytes);
  const dataset = Uint8Array.from(event.datasetBytes);
  return Object.freeze({
    index: event.index,
    get parametersBytes() { return Uint8Array.from(parameters); },
    get datasetBytes() { return Uint8Array.from(dataset); },
  });
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

function freezeGeneration(generation) {
  return Object.freeze({
    ...generation,
    attempt: Object.freeze({ ...generation.attempt }),
  });
}

function freezeConditioning(conditioning) {
  return Object.freeze({
    ...conditioning,
    attempt: Object.freeze({ ...conditioning.attempt }),
  });
}

function freeze(state) {
  return Object.freeze({
    ...state,
    documents: Object.freeze({ ...state.documents }),
    compile: Object.freeze({ ...state.compile }),
    run: Object.freeze({ ...state.run }),
    generation: freezeGeneration(state.generation),
    conditioning: freezeConditioning(state.conditioning),
    artifacts: Object.freeze([...state.artifacts]),
  });
}
