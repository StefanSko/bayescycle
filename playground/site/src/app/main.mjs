import { parseDocument, serializeDocument } from "../data/documents.mjs";
import { parseGeneratedDatasets } from "../generation/artifact.mjs";
import {
  fixed,
  generateDatasets,
  generationInvalidationKey,
  modelPrior,
  posteriorOf,
} from "../generation/plan.mjs";
import { readDashboardData, renderEssRhat, renderPrecis, renderTrank } from "../dashboard/index.mjs";
import { BrowserRuntime } from "../runtime/browser-runtime.mjs";
import { renderRecoverySummary } from "./recovery.mjs";
import { decodeProject, encodeProject, FRAGMENT_WARN_LENGTH } from "./share.mjs";
import { initialState, reduce } from "./state.mjs";

const runtime = new BrowserRuntime();
let state = initialState();
let revision = 0;
let pendingSharedProject;
let exampleLoadRevision = 0;
const examples = new Map();
const objectUrls = new Set();
const EXAMPLES_ROOT = new URL("../../examples/", import.meta.url);

const source = element("#model-source");
const observed = element("#observed-data");
const design = element("#design-data");
const truth = element("#truth-data");

source.addEventListener("input", () => {
  exampleLoadRevision += 1;
  element("#progress").replaceChildren();
  dispatch({ type: "source-edited", source: source.value, revision: ++revision });
});
observed.addEventListener("input", observedEdited);
for (const input of [design, truth]) input.addEventListener("input", generationInputsEdited);
for (const selector of [
  "#chains", "#warmup", "#draws", "#inference-seed", "#target-accept",
  "#max-treedepth",
]) {
  element(selector).addEventListener("input", settingsEdited);
}
for (const selector of ["#generation-seed", "#generation-count"]) {
  element(selector).addEventListener("input", generationSettingsEdited);
}
for (const radio of document.querySelectorAll("input[name='param-source']")) {
  radio.addEventListener("change", () => {
    dispatch({ type: "parameter-source-edited", revision: ++revision });
  });
}
for (const radio of document.querySelectorAll("input[name='dataset-source']")) {
  radio.addEventListener("change", () => {
    dispatch({
      type: "dataset-source-edited", source: selectedDatasetSource(), revision: ++revision,
    });
  });
}
element("#compile-button").addEventListener("click", () => void compileModel());
element("#generate-button").addEventListener("click", () => void generateCollection());
element("#generated-dataset-index").addEventListener("change", () => {
  selectGeneratedPair(integerValue("#generated-dataset-index"));
});
element("#fit-button").addEventListener("click", () => {
  launchRun(selectedDatasetSource() === "generated" ? sampleGenerated : samplePosterior);
});
element("#examples-menu").addEventListener("change", () => void loadExample());
element("#share-button").addEventListener("click", () => void shareProject());
element("#load-shared").addEventListener("click", loadSharedProject);

async function initializeVersion() {
  try {
    const response = await fetch(new URL("../../VERSION.json", import.meta.url));
    if (!response.ok) return;
    const value = await response.json();
    if (typeof value.version === "string") element("#playground-version").textContent = `v${value.version}`;
  } catch {
    // The static fallback remains visible when the version sidecar is unavailable.
  }
}

async function initializeExamples() {
  try {
    const response = await fetch(new URL("EXAMPLES.json", EXAMPLES_ROOT));
    if (!response.ok) throw new Error("Could not load examples");
    for (const entry of await response.json()) {
      examples.set(entry.id, entry);
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.label;
      element("#examples-menu").append(option);
    }
  } catch (error) {
    element("#compile-error").hidden = false;
    element("#compile-error").textContent = message(error);
  }
}

async function loadExample() {
  const selectedId = element("#examples-menu").value;
  const entry = examples.get(selectedId);
  if (entry === undefined) return;
  const loadRevision = ++exampleLoadRevision;
  const [modelSource, observedText, designText, truthText] = await Promise.all([
    fetchAsset(entry.source), fetchOptionalAsset(entry.observed),
    fetchOptionalAsset(entry.design), fetchOptionalAsset(entry.truth),
  ]);
  if (loadRevision !== exampleLoadRevision || element("#examples-menu").value !== selectedId) return;
  setProject({ source: modelSource, observed: observedText, design: designText, truth: truthText });
}

async function fetchAsset(path) {
  const response = await fetch(new URL(path, EXAMPLES_ROOT));
  if (!response.ok) throw new Error(`Could not load example asset ${path}`);
  return response.text();
}

function fetchOptionalAsset(path) {
  return path === undefined ? Promise.resolve("") : fetchAsset(path);
}

async function shareProject() {
  const project = {
    v: 1,
    source: source.value,
    observed: observed.value,
    design: design.value,
    truth: truth.value,
    sampler: samplerSettings(),
    generation: {
      seed: integerValue("#generation-seed"),
      count: integerValue("#generation-count"),
    },
  };
  const payload = await encodeProject(project);
  const url = new URL(window.location.href);
  url.hash = `project=${payload}`;
  element("#share-url").value = url.href;
  element("#share-output").hidden = false;
  element("#share-warning").hidden = payload.length <= FRAGMENT_WARN_LENGTH;
}

async function initializeSharedProject() {
  if (!window.location.hash.startsWith("#project=")) return;
  try {
    pendingSharedProject = await decodeProject(window.location.hash.slice(9));
    element("#share-source").textContent = String(pendingSharedProject.source ?? "");
    element("#share-review").hidden = false;
  } catch (error) {
    element("#share-review").hidden = false;
    element("#share-source").textContent = message(error);
    element("#load-shared").disabled = true;
  }
}

function loadSharedProject() {
  if (pendingSharedProject === undefined) return;
  setProject({
    source: String(pendingSharedProject.source ?? ""),
    observed: String(pendingSharedProject.observed ?? ""),
    design: String(pendingSharedProject.design ?? ""),
    truth: String(pendingSharedProject.truth ?? ""),
  });
  applySamplerSettings(pendingSharedProject.sampler);
  applyGenerationSettings(pendingSharedProject.generation, pendingSharedProject.sampler);
  element("#share-review").hidden = true;
}

function setProject(project) {
  exampleLoadRevision += 1;
  element("#progress").replaceChildren();
  for (const [control, value] of [[source, project.source], [observed, project.observed], [design, project.design], [truth, project.truth]]) {
    control.textContent = value;
    control.value = value;
  }
  dispatch({ type: "source-edited", source: project.source, revision: ++revision });
  dispatch({ type: "documents-edited", documents: { observed: project.observed, design: project.design, truth: project.truth }, revision: ++revision });
  element("#generation-documents").open = project.design.trim() !== "";
  element("#share-output").hidden = true;
}

function applySamplerSettings(settings) {
  if (settings === null || typeof settings !== "object") return;
  const fields = { chains: "#chains", num_warmup: "#warmup", num_draws: "#draws", seed: "#inference-seed", target_accept: "#target-accept", max_treedepth: "#max-treedepth" };
  let applied = false;
  for (const [name, selector] of Object.entries(fields)) {
    if (typeof settings[name] === "number") {
      element(selector).value = String(settings[name]);
      applied = true;
    }
  }
  if (applied) settingsEdited();
}

function applyGenerationSettings(settings, legacySampler) {
  const generation = settings !== null && typeof settings === "object" ? settings : {};
  const legacy = legacySampler !== null && typeof legacySampler === "object" ? legacySampler : {};
  const seed = typeof generation.seed === "number"
    ? generation.seed
    : typeof legacy.seed === "number" ? legacy.seed : undefined;
  const count = typeof generation.count === "number"
    ? generation.count
    : typeof generation.num_draws === "number"
      ? generation.num_draws
      : typeof legacy.num_draws === "number" ? legacy.num_draws : undefined;
  let applied = false;
  for (const [value, selector] of [[seed, "#generation-seed"], [count, "#generation-count"]]) {
    if (value !== undefined) {
      element(selector).value = String(value);
      applied = true;
    }
  }
  if (applied) generationSettingsEdited();
}

function launchRun(operation) {
  void operation().catch((error) => {
    element("#run-error").hidden = false;
    element("#run-error").textContent = message(error);
  });
}

function settingsEdited() {
  element("#progress").replaceChildren();
  dispatch({ type: "inference-settings-edited", revision: ++revision });
}

function generationSettingsEdited() {
  element("#progress").replaceChildren();
  dispatch({ type: "generation-settings-scoped-edited", revision: ++revision });
}

function observedEdited() {
  exampleLoadRevision += 1;
  element("#progress").replaceChildren();
  dispatch({
    type: "observed-input-edited",
    documents: { ...state.documents, observed: observed.value },
    revision: ++revision,
  });
}

function generationInputsEdited() {
  exampleLoadRevision += 1;
  element("#progress").replaceChildren();
  dispatch({
    type: "generation-input-edited",
    documents: { ...state.documents, design: design.value, truth: truth.value },
    revision: ++revision,
  });
}

async function compileModel() {
  const requestId = crypto.randomUUID();
  const sourceRevision = state.sourceRevision;
  dispatch({ type: "compile-started", requestId, revision: sourceRevision });
  try {
    const result = await runtime.compile(state.source);
    if (result.ok) {
      dispatch({ type: "compile-succeeded", requestId, revision: sourceRevision, irBytes: result.irBytes, irHash: result.irHash });
    } else {
      dispatch({ type: "compile-failed", requestId, revision: sourceRevision, error: result.traceback });
    }
  } catch (error) {
    dispatch({ type: "compile-failed", requestId, revision: sourceRevision, error: message(error) });
  }
}

async function samplePosterior() {
  const dataBytes = documentBytes(observed.value);
  await sampleData(dataBytes, "observed");
}

async function generateCollection() {
  if (state.compile.status !== "compiled" || state.run.status === "running" ||
      state.generation.attempt.status === "running") return;
  const modelBytes = compiledBytes();
  const designBytes = documentBytes(design.value);
  const sourceKind = selectedParamSource();
  let parameterSource;
  let fixedParametersBytes;
  let sourceFitLineageKey;
  if (sourceKind === "fixed") {
    fixedParametersBytes = documentBytes(truth.value);
    parameterSource = fixed(fixedParametersBytes);
  } else if (sourceKind === "prior") {
    parameterSource = modelPrior(modelBytes);
  } else {
    const fit = state.conditioning.fit;
    if (fit === null || fit.fitArtifact === undefined) return;
    parameterSource = posteriorOf(fit.fitArtifact);
    sourceFitLineageKey = fit.lineageKey;
  }
  const settings = generationSettings();
  const plan = generateDatasets(modelBytes, {
    design: designBytes,
    parameterSource,
    count: settings.count,
    seed: settings.seed,
  });
  const guard = {
    compileRevision: state.compile.revision,
    inputRevision: state.generation.inputRevision,
    settingsRevision: state.generation.settingsRevision,
    sourceKind: sourceKind === "prior" ? "model-prior" : sourceKind,
    fitLineageKey: sourceFitLineageKey ?? null,
  };
  const dependencyKey = await generationInvalidationKey(plan);
  const requestId = crypto.randomUUID();
  dispatch({ type: "generation-started", requestId, dependencyKey, guard });
  if (state.generation.attempt.requestId !== requestId) return;
  try {
    const result = await runtime.run({
      type: "run", id: requestId, operation: "generate", plan,
    });
    const artifact = result.artifacts.find(
      (entry) => entry.name === "generated_datasets.ndjson",
    );
    if (artifact === undefined) throw new Error("Runtime returned no generated collection");
    const parsed = parseGeneratedDatasets(artifact.bytes);
    const selected = parsed.select(0);
    dispatch({
      type: "generation-succeeded", requestId, dependencyKey,
      collection: {
        sourceKind: sourceKind === "prior" ? "model-prior" : sourceKind,
        ...(sourceFitLineageKey === undefined ? {} : { sourceFitLineageKey }),
        plan, artifact, artifacts: result.artifacts, parsed,
      },
      selection: {
        revision: ++revision,
        index: selected.drawIndex,
        parametersBytes: selected.parametersBytes,
        datasetBytes: selected.datasetBytes,
      },
    });
  } catch (error) {
    dispatch({
      type: "generation-failed", requestId, dependencyKey, error: message(error),
    });
  }
}

function selectGeneratedPair(index) {
  const collection = state.generation.collection;
  if (collection === null) return;
  const selected = collection.parsed.select(index);
  dispatch({
    type: "selection-edited", revision: ++revision, index: selected.drawIndex,
    parametersBytes: selected.parametersBytes, datasetBytes: selected.datasetBytes,
  });
}

async function sampleGenerated() {
  const selected = state.generation.selected;
  if (selected === null) return;
  await sampleData(selected.datasetBytes, "generated", selected.parametersBytes);
}

async function sampleData(dataBytes, datasetSource, recoveryTruth) {
  if (state.compile.status !== "compiled" || state.run.status === "running") return;
  if (state.conditioning.datasetSource !== datasetSource) {
    dispatch({ type: "dataset-source-edited", source: datasetSource, revision: ++revision });
  }
  const requestId = crypto.randomUUID();
  const projectRevision = state.projectRevision;
  const settings = sampleSettings();
  const guard = {
    compileRevision: state.compile.revision,
    settingsRevision: state.conditioning.settingsRevision,
    datasetSource,
    datasetSourceRevision: state.conditioning.datasetSourceRevision,
    observed: state.documents.observed,
    selectionRevision: state.generation.selectionRevision,
  };
  const dependencyKey = await conditioningDependencyKey(
    state.compile.irBytes,
    dataBytes,
    settings,
  );
  if (selectedDatasetSource() !== datasetSource) return;
  element("#progress").replaceChildren();
  dispatch({
    type: "conditioning-started", requestId, dependencyKey, datasetSource, guard,
  });
  if (state.conditioning.attempt.requestId !== requestId) return;
  dispatch({
    type: "run-started",
    requestId,
    revision: projectRevision,
    operation: "condition",
    datasetSource,
  });
  try {
    const sampled = await runtime.run({
      type: "run",
      id: requestId,
      operation: "condition",
      modelIr: state.compile.irBytes,
      data: dataBytes,
      settings,
      ...(recoveryTruth === undefined ? {} : { pairedParameters: recoveryTruth }),
    }, (event) => renderActiveProgress(requestId, projectRevision, event));
    const posterior = sampled.artifacts.find((artifact) => artifact.name === "posterior.ndjson");
    if (posterior === undefined) throw new Error("Runtime returned no posterior artifact");
    const artifacts = [...sampled.artifacts];
    const lineageKey = await bytesHash(posterior.bytes);
    dispatch({
      type: "conditioning-committed",
      requestId,
      dependencyKey,
      revision: projectRevision,
      artifacts,
      notice: sampled.notice,
      fit: {
        datasetSource,
        lineageKey,
        artifacts,
        fitArtifact: sampled.fitArtifact,
      },
    });
  } catch (error) {
    dispatch({ type: "run-failed", requestId, revision: projectRevision, error: message(error) });
    dispatch({
      type: "conditioning-failed", requestId, dependencyKey, error: message(error),
    });
  }
}

function compiledBytes() {
  if (state.compile.status !== "compiled") throw new Error("Compile the model first");
  return state.compile.irBytes;
}

function documentBytes(text) {
  return new TextEncoder().encode(serializeDocument(parseDocument(text)));
}

async function bytesHash(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return `sha256:${[...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

async function conditioningDependencyKey(modelBytes, dataBytes, settings) {
  const framed = new TextEncoder().encode(JSON.stringify({
    model: await bytesHash(modelBytes),
    data: await bytesHash(dataBytes),
    settings,
  }));
  return bytesHash(framed);
}

function sampleSettings() {
  const settings = samplerSettings();
  if (!validSampleSettings(settings)) {
    throw new Error("Sampling requires at least 1 chain and 4 retained draws with valid settings");
  }
  return settings;
}

function generationSettings() {
  const seed = integerValue("#generation-seed");
  const count = integerValue("#generation-count");
  if (!validGenerationSeed(seed)) {
    throw new Error("Generation seed must be a nonnegative safe integer");
  }
  if (!validGenerationCount(count)) {
    throw new Error("Generation count must be an integer in 1..1000");
  }
  return { seed, count };
}

function validGenerationSeed(seed = integerValue("#generation-seed")) {
  return Number.isSafeInteger(seed) && seed >= 0;
}

function validGenerationCount(count = integerValue("#generation-count")) {
  return Number.isSafeInteger(count) && count >= 1 && count <= 1000;
}

function validSeedSettings(settings = samplerSettings()) {
  return Number.isSafeInteger(settings.seed) && settings.seed >= 0;
}

function validSampleSettings(settings = samplerSettings()) {
  return Number.isSafeInteger(settings.chains) && settings.chains >= 1 &&
    Number.isSafeInteger(settings.num_warmup) && settings.num_warmup >= 0 &&
    Number.isSafeInteger(settings.num_draws) && settings.num_draws >= 4 &&
    validSeedSettings(settings) &&
    Number.isSafeInteger(settings.max_treedepth) && settings.max_treedepth >= 1 &&
    Number.isFinite(settings.target_accept) && settings.target_accept > 0 &&
    settings.target_accept < 1;
}

function samplerSettings() {
  return {
    chains: integerValue("#chains"),
    num_warmup: integerValue("#warmup"),
    num_draws: integerValue("#draws"),
    seed: integerValue("#inference-seed"),
    target_accept: numberValue("#target-accept"),
    max_treedepth: integerValue("#max-treedepth"),
  };
}

function renderActiveProgress(requestId, projectRevision, event) {
  if (state.run.status !== "running" || state.run.requestId !== requestId ||
      state.run.revision !== projectRevision) return;
  renderProgress(event);
}

function renderProgress(event) {
  const lineId = `progress-chain-${event.chainId}`;
  let line = document.getElementById(lineId);
  if (line === null) {
    line = document.createElement("div");
    line.id = lineId;
    element("#progress").append(line);
  }
  line.textContent = `chain ${event.chainId + 1}: ${event.retainedDraws} draws, ${event.divergences} divergences`;
}

function dispatch(event) {
  state = reduce(state, event);
  render();
}

function render() {
  const hash = element("#ir-hash");
  hash.hidden = state.compile.status !== "compiled";
  hash.textContent = state.compile.status === "compiled" ? `sha256:${state.compile.irHash}` : "";
  element("#compile-status").textContent = state.compile.status === "compiling"
    ? "Compiling in an isolated worker…"
    : state.compile.status === "compiled" ? "Model compiled successfully." : "";
  const compileError = element("#compile-error");
  compileError.hidden = state.compile.status !== "failed";
  compileError.textContent = state.compile.status === "failed" ? state.compile.error : "";
  element("#compile-button").disabled = state.compile.status === "compiling" ||
    state.run.status === "running" || state.source.trim() === "";
  element("#share-button").disabled = state.source.trim() === "";
  const visibleArtifacts = [
    ...state.artifacts,
    ...(state.generation.collection?.artifacts ?? []),
  ];
  const hasArtifact = (name) => visibleArtifacts.some((artifact) => artifact.name === name);
  const posteriorAvailable = hasArtifact("posterior.ndjson") && hasArtifact("data.json");
  const posteriorSource = element("#param-source-posterior");
  posteriorSource.disabled = !posteriorAvailable;
  element("#posterior-source-hint").hidden = posteriorAvailable;
  if (posteriorSource.checked && posteriorSource.disabled) element("#param-source-fixed").checked = true;

  renderGenerationSelection();
  const generatedAvailable = state.generation.selected !== null;
  const generatedSource = element("#dataset-source-generated");
  generatedSource.disabled = !generatedAvailable;
  if (generatedSource.checked && generatedSource.disabled) element("#dataset-source-observed").checked = true;

  const paramSource = selectedParamSource();
  const datasetSource = selectedDatasetSource();
  element("#fixed-values-field").hidden = paramSource !== "fixed";
  element("#generate-button").textContent = {
    fixed: "Generate at fixed values",
    prior: "Generate from model prior",
    posterior: "Generate from posterior",
  }[paramSource];
  element("#fit-button").textContent = datasetSource === "generated"
    ? "Fit generated dataset"
    : "Fit observed data";

  const unavailable = state.compile.status !== "compiled" || state.run.status === "running" ||
    state.generation.attempt.status === "running";
  element("#generate-button").disabled = unavailable || !validGenerationSeed() ||
    !validGenerationCount() ||
    design.value.trim() === "" ||
    (paramSource === "fixed" && truth.value.trim() === "") ||
    (paramSource === "posterior" && !posteriorAvailable);
  element("#fit-button").disabled = unavailable || !validSampleSettings() ||
    (datasetSource === "observed" && observed.value.trim() === "") ||
    (datasetSource === "generated" && !generatedAvailable);
  element("#run-status").textContent = state.generation.attempt.status === "running"
    ? "Generating paired datasets is running…"
    : state.run.status === "running" ? `${operationLabel(state.run.operation)} is running…` : "";

  const runError = element("#run-error");
  if (state.generation.attempt.status === "failed") {
    runError.hidden = false;
    runError.textContent = state.generation.attempt.error;
  } else if (state.run.status === "failed") {
    runError.hidden = false;
    runError.textContent = state.run.error;
  } else if (state.notice !== null) {
    runError.hidden = false;
    runError.textContent = state.notice;
  } else {
    runError.hidden = true;
    runError.textContent = "";
  }
  renderArtifacts(visibleArtifacts);
}

function renderGenerationSelection() {
  const container = element("#generation-selection");
  const selector = element("#generated-dataset-index");
  const summary = element("#selected-pair-summary");
  const collection = state.generation.collection;
  if (collection === null) {
    container.hidden = true;
    selector.disabled = true;
    selector.replaceChildren();
    summary.textContent = "";
    return;
  }
  if (selector.options.length !== collection.parsed.count) {
    selector.replaceChildren(...Array.from({ length: collection.parsed.count }, (_, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `Dataset ${index + 1}`;
      return option;
    }));
  }
  const selected = state.generation.selected;
  selector.disabled = selected === null;
  if (selected !== null) selector.value = String(selected.index);
  summary.textContent = selected === null
    ? "Select a generated parameter/dataset pair."
    : `Selected dataset ${selected.index + 1} of ${collection.parsed.count} with its paired parameters.`;
  container.hidden = false;
}

function renderArtifacts(artifacts) {
  for (const url of objectUrls) URL.revokeObjectURL(url);
  objectUrls.clear();
  const mapping = {
    "model.ir.json": "#artifact-model",
    "data.json": "#artifact-data",
    "posterior.ndjson": "#artifact-posterior",
    "diagnostics.json": "#artifact-diagnostics",
    "recovery_check.json": "#artifact-recovery",
    "generated_datasets.ndjson": "#artifact-generated-datasets",
    "generation-plan.json": "#artifact-generation-plan",
    "run.json": "#artifact-generation-run",
    "design.json": "#artifact-generation-design",
    "fixed-parameters.json": "#artifact-fixed-parameters",
    "source-posterior.ndjson": "#artifact-source-posterior",
    "source-fit-data.json": "#artifact-source-fit-data",
  };
  for (const selector of Object.values(mapping)) element(selector).hidden = true;
  for (const artifact of artifacts) {
    const selector = mapping[artifact.name];
    if (selector === undefined) continue;
    const item = element(selector);
    const url = URL.createObjectURL(new Blob([artifact.bytes], { type: artifact.mediaType }));
    objectUrls.add(url);
    item.querySelector("a").href = url;
    item.hidden = false;
  }
  element("#artifacts").hidden = artifacts.length === 0;
  renderRecovery(artifacts);
  renderPlots(artifacts);
}

function renderRecovery(artifacts) {
  const recovery = artifacts.find((artifact) => artifact.name === "recovery_check.json");
  const summary = element("#recovery-summary");
  if (recovery === undefined) {
    summary.hidden = true;
    summary.querySelector("div").replaceChildren();
    return;
  }
  const report = JSON.parse(new TextDecoder().decode(recovery.bytes));
  summary.querySelector("div").innerHTML = renderRecoverySummary(report);
  summary.hidden = false;
}

function renderPlots(artifacts) {
  const posterior = artifacts.find((artifact) => artifact.name === "posterior.ndjson");
  const diagnostics = artifacts.find((artifact) => artifact.name === "diagnostics.json");
  const plots = element("#plots");
  const plotGrid = element("#plot-grid");
  if (posterior === undefined || diagnostics === undefined) {
    plotGrid.hidden = true;
    plots.hidden = !artifacts.some((artifact) => artifact.name === "recovery_check.json");
    for (const selector of ["#plot-trank", "#plot-ess-rhat", "#plot-precis"]) {
      element(selector).replaceChildren();
    }
    return;
  }
  try {
    const data = readDashboardData({ fits: [posterior.bytes], diagnose: diagnostics.bytes });
    element("#plot-trank").innerHTML = renderTrank(data);
    element("#plot-ess-rhat").innerHTML = renderEssRhat(data);
    element("#plot-precis").innerHTML = renderPrecis(data);
    plotGrid.hidden = false;
    plots.hidden = false;
  } catch (error) {
    plotGrid.hidden = true;
    plots.hidden = !artifacts.some((artifact) => artifact.name === "recovery_check.json");
    element("#run-error").hidden = false;
    element("#run-error").textContent = `Artifacts downloaded, but plots could not render: ${message(error)}`;
  }
}

function operationLabel(operation) {
  const labels = {
    condition: "Fitting",
  };
  return labels[operation] ?? "Operation";
}

function selectedParamSource() {
  return element("input[name='param-source']:checked").value;
}

function selectedDatasetSource() {
  return element("input[name='dataset-source']:checked").value;
}

function integerValue(selector) { return Number(element(selector).value); }
function numberValue(selector) { return Number(element(selector).value); }
function element(selector) {
  const value = document.querySelector(selector);
  if (value === null) throw new Error(`Missing required element ${selector}`);
  return value;
}
function message(error) { return error instanceof Error ? error.message : String(error); }

render();
void initializeVersion();
void initializeExamples();
void initializeSharedProject();
