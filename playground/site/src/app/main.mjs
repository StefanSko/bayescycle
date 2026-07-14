import { parseDocument, serializeDocument } from "../data/documents.mjs";
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
for (const input of [observed, design, truth]) input.addEventListener("input", documentsEdited);
for (const selector of [
  "#chains", "#warmup", "#draws", "#inference-seed", "#target-accept",
  "#max-treedepth",
]) {
  element(selector).addEventListener("input", settingsEdited);
}
element("#generation-seed").addEventListener("input", generationSettingsEdited);
for (const selector of ["input[name='param-source']", "input[name='dataset-source']"]) {
  for (const radio of document.querySelectorAll(selector)) radio.addEventListener("change", render);
}
element("#compile-button").addEventListener("click", () => void compileModel());
element("#generate-button").addEventListener("click", () => {
  const operations = {
    fixed: simulateData,
    prior: runPriorPredictive,
    posterior: runPosteriorPredictive,
  };
  launchRun(operations[selectedParamSource()]);
});
element("#fit-button").addEventListener("click", () => {
  launchRun(selectedDatasetSource() === "generated" ? sampleSimulated : samplePosterior);
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
    generation: { seed: integerValue("#generation-seed") },
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
  applyGenerationSettings(pendingSharedProject.generation);
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

function applyGenerationSettings(settings) {
  if (settings === null || typeof settings !== "object") return;
  if (typeof settings.seed === "number") {
    element("#generation-seed").value = String(settings.seed);
    generationSettingsEdited();
  }
}

function launchRun(operation) {
  void operation().catch((error) => {
    element("#run-error").hidden = false;
    element("#run-error").textContent = message(error);
  });
}

function settingsEdited() {
  element("#progress").replaceChildren();
  dispatch({ type: "settings-edited", revision: ++revision });
}

function generationSettingsEdited() {
  element("#progress").replaceChildren();
  dispatch({ type: "generation-settings-edited", revision: ++revision });
}

function documentsEdited() {
  exampleLoadRevision += 1;
  element("#progress").replaceChildren();
  dispatch({
    type: "documents-edited",
    documents: { observed: observed.value, design: design.value, truth: truth.value },
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
  await sampleData(dataBytes);
}

async function runPriorPredictive() {
  await runSingle({
    operation: "prior-predictive",
    modelIr: compiledBytes(),
    data: documentBytes(design.value),
    settings: generationSettings(),
  });
}

async function runPosteriorPredictive() {
  const posterior = state.artifacts.find((artifact) => artifact.name === "posterior.ndjson");
  const data = state.artifacts.find((artifact) => artifact.name === "data.json");
  if (posterior === undefined || data === undefined) return;
  await runSingle({
    operation: "posterior-predictive",
    modelIr: compiledBytes(),
    data: data.bytes,
    fit: posterior.bytes,
    settings: generationSettings(),
  });
}

async function simulateData() {
  await runSingle({
    operation: "simulate",
    modelIr: compiledBytes(),
    data: documentBytes(design.value),
    truth: documentBytes(truth.value),
    settings: generationSettings(),
  });
}

async function sampleSimulated() {
  const simulated = state.artifacts.find((artifact) => artifact.name === "simulated_data.json");
  if (simulated === undefined) return;
  await sampleData(simulated.bytes, documentBytes(truth.value));
}

async function sampleData(dataBytes, recoveryTruth) {
  if (state.compile.status !== "compiled" || state.run.status === "running") return;
  const requestId = crypto.randomUUID();
  const projectRevision = state.projectRevision;
  element("#progress").replaceChildren();
  dispatch({ type: "run-started", requestId, revision: projectRevision, operation: "sample" });
  try {
    const sampled = await runtime.run({
      type: "run",
      id: requestId,
      operation: "sample",
      modelIr: state.compile.irBytes,
      data: dataBytes,
      settings: sampleSettings(),
    }, (event) => renderActiveProgress(requestId, projectRevision, event));
    const posterior = sampled.artifacts.find((artifact) => artifact.name === "posterior.ndjson");
    if (posterior === undefined) throw new Error("Runtime returned no posterior artifact");
    let artifacts = [...sampled.artifacts];
    const warnings = [];
    try {
      const diagnosed = await runtime.run({ type: "run", id: crypto.randomUUID(), operation: "diagnose", fit: posterior.bytes });
      artifacts = [...artifacts, ...diagnosed.artifacts];
    } catch (error) {
      warnings.push(`Posterior completed; diagnostics unavailable: ${message(error)}`);
    }
    if (recoveryTruth !== undefined) {
      try {
        const recovery = await runtime.run({ type: "run", id: crypto.randomUUID(), operation: "recover-check", fit: posterior.bytes, truth: recoveryTruth });
        artifacts = [...artifacts, ...recovery.artifacts];
      } catch (error) {
        warnings.push(`Posterior completed; recovery check unavailable: ${message(error)}`);
      }
    }
    dispatch({
      type: "run-succeeded",
      requestId,
      revision: projectRevision,
      artifacts,
      notice: warnings.length === 0 ? null : warnings.join("\n"),
    });
  } catch (error) {
    dispatch({ type: "run-failed", requestId, revision: projectRevision, error: message(error) });
  }
}

async function runSingle(request) {
  if (state.compile.status !== "compiled" || state.run.status === "running") return;
  const requestId = crypto.randomUUID();
  const projectRevision = state.projectRevision;
  element("#progress").replaceChildren();
  dispatch({ type: "run-started", requestId, revision: projectRevision, operation: request.operation });
  try {
    const result = await runtime.run(
      { type: "run", id: requestId, ...request },
      (event) => renderActiveProgress(requestId, projectRevision, event),
    );
    dispatch({ type: "run-succeeded", requestId, revision: projectRevision, artifacts: result.artifacts });
  } catch (error) {
    dispatch({ type: "run-failed", requestId, revision: projectRevision, error: message(error) });
  }
}

function compiledBytes() {
  if (state.compile.status !== "compiled") throw new Error("Compile the model first");
  return state.compile.irBytes;
}

function documentBytes(text) {
  return new TextEncoder().encode(serializeDocument(parseDocument(text)));
}

function sampleSettings() {
  const settings = samplerSettings();
  if (!validSampleSettings(settings)) {
    throw new Error("Sampling requires at least 1 chain and 4 retained draws with valid settings");
  }
  return settings;
}

function generationSettings() {
  const settings = { ...samplerSettings(), seed: integerValue("#generation-seed") };
  if (!validGenerationSettings(settings)) {
    throw new Error("Generation seed must be a nonnegative safe integer");
  }
  return settings;
}

function validGenerationSettings(settings = { seed: integerValue("#generation-seed") }) {
  return Number.isSafeInteger(settings.seed) && settings.seed >= 0;
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
  const hasArtifact = (name) => state.artifacts.some((artifact) => artifact.name === name);
  const posteriorAvailable = hasArtifact("posterior.ndjson") && hasArtifact("data.json");
  const posteriorSource = element("#param-source-posterior");
  posteriorSource.disabled = !posteriorAvailable;
  element("#posterior-source-hint").hidden = posteriorAvailable;
  if (posteriorSource.checked && posteriorSource.disabled) element("#param-source-fixed").checked = true;

  const generatedAvailable = hasArtifact("simulated_data.json") && truth.value.trim() !== "";
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

  const unavailable = state.compile.status !== "compiled" || state.run.status === "running";
  element("#generate-button").disabled = unavailable || !validGenerationSettings() ||
    (paramSource !== "posterior" && design.value.trim() === "") ||
    (paramSource === "fixed" && truth.value.trim() === "") ||
    (paramSource === "posterior" && !posteriorAvailable);
  element("#fit-button").disabled = unavailable || !validSampleSettings() ||
    (datasetSource === "observed" && observed.value.trim() === "") ||
    (datasetSource === "generated" && !generatedAvailable);
  element("#run-status").textContent = state.run.status === "running"
    ? `${operationLabel(state.run.operation)} is running…`
    : "";

  const runError = element("#run-error");
  if (state.run.status === "failed") {
    runError.hidden = false;
    runError.textContent = state.run.error;
  } else if (state.notice !== null) {
    runError.hidden = false;
    runError.textContent = state.notice;
  } else {
    runError.hidden = true;
    runError.textContent = "";
  }
  renderArtifacts(state.artifacts);
}

function renderArtifacts(artifacts) {
  for (const url of objectUrls) URL.revokeObjectURL(url);
  objectUrls.clear();
  const mapping = {
    "model.ir.json": "#artifact-model",
    "data.json": "#artifact-data",
    "posterior.ndjson": "#artifact-posterior",
    "diagnostics.json": "#artifact-diagnostics",
    "prior_predictive.ndjson": "#artifact-prior",
    "posterior_predictive.ndjson": "#artifact-posterior-predictive",
    "simulated_data.json": "#artifact-simulated",
    "recovery_check.json": "#artifact-recovery",
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
    sample: "Fitting",
    "prior-predictive": "Generating from model prior",
    "posterior-predictive": "Generating from posterior",
    simulate: "Generating at fixed values",
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
