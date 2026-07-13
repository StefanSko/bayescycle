import { compile } from "../compile/index.mjs";
import { parseDocument, serializeDocument } from "../data/documents.mjs";
import { readDashboardData, renderEssRhat, renderPrecis, renderTrank } from "../dashboard/index.mjs";
import { BrowserRuntime } from "../runtime/browser-runtime.mjs";
import { decodeProject, encodeProject, FRAGMENT_WARN_LENGTH } from "./share.mjs";
import { initialState, reduce } from "./state.mjs";

const runtime = new BrowserRuntime();
let state = initialState();
let revision = 0;
let pendingSharedProject;
const examples = new Map();
const objectUrls = new Set();
const EXAMPLES_ROOT = new URL("../../examples/", import.meta.url);

const source = element("#model-source");
const observed = element("#observed-data");
const design = element("#design-data");
const truth = element("#truth-data");

source.addEventListener("input", () => {
  element("#progress").replaceChildren();
  dispatch({ type: "source-edited", source: source.value, revision: ++revision });
});
for (const input of [observed, design, truth]) input.addEventListener("input", documentsEdited);
element("#compile-button").addEventListener("click", () => void compileModel());
element("#sample-button").addEventListener("click", () => void samplePosterior());
element("#prior-button").addEventListener("click", () => void runPriorPredictive());
element("#simulate-button").addEventListener("click", () => void simulateData());
element("#sample-simulated-button").addEventListener("click", () => void sampleSimulated());
element("#examples-menu").addEventListener("change", () => void loadExample());
element("#share-button").addEventListener("click", () => void shareProject());
element("#load-shared").addEventListener("click", loadSharedProject);

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
  const entry = examples.get(element("#examples-menu").value);
  if (entry === undefined) return;
  const [modelSource, observedText, designText, truthText] = await Promise.all([
    fetchAsset(entry.source), fetchOptionalAsset(entry.observed),
    fetchOptionalAsset(entry.design), fetchOptionalAsset(entry.truth),
  ]);
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
  element("#share-review").hidden = true;
}

function setProject(project) {
  for (const [control, value] of [[source, project.source], [observed, project.observed], [design, project.design], [truth, project.truth]]) {
    control.textContent = value;
    control.value = value;
  }
  dispatch({ type: "source-edited", source: project.source, revision: ++revision });
  dispatch({ type: "documents-edited", documents: { observed: project.observed, design: project.design, truth: project.truth }, revision: ++revision });
  element("#simulation-documents").open = project.design.trim() !== "" || project.truth.trim() !== "";
  element("#share-output").hidden = true;
}

function applySamplerSettings(settings) {
  if (settings === null || typeof settings !== "object") return;
  const fields = { chains: "#chains", num_warmup: "#warmup", num_draws: "#draws", seed: "#seed", target_accept: "#target-accept", max_treedepth: "#max-treedepth" };
  for (const [name, selector] of Object.entries(fields)) {
    if (typeof settings[name] === "number") element(selector).value = String(settings[name]);
  }
}

function documentsEdited() {
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
    const result = await compile(state.source);
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
    settings: samplerSettings(),
  });
}

async function simulateData() {
  await runSingle({
    operation: "simulate",
    modelIr: compiledBytes(),
    data: documentBytes(design.value),
    truth: documentBytes(truth.value),
    settings: samplerSettings(),
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
  dispatch({ type: "run-started", requestId, revision: projectRevision });
  try {
    const sampled = await runtime.run({
      operation: "sample",
      modelIr: state.compile.irBytes,
      data: dataBytes,
      settings: samplerSettings(),
    }, renderProgress);
    const posterior = sampled.artifacts.find((artifact) => artifact.name === "posterior.ndjson");
    if (posterior === undefined) throw new Error("Runtime returned no posterior artifact");
    let artifacts = [...sampled.artifacts];
    try {
      const diagnosed = await runtime.run({ operation: "diagnose", fit: posterior.bytes });
      artifacts = [...artifacts, ...diagnosed.artifacts];
    } catch (error) {
      showFollowupError("diagnostics", error);
    }
    if (recoveryTruth !== undefined) {
      try {
        const recovery = await runtime.run({ operation: "recover-check", fit: posterior.bytes, truth: recoveryTruth });
        artifacts = [...artifacts, ...recovery.artifacts];
      } catch (error) {
        showFollowupError("recovery check", error);
      }
    }
    dispatch({ type: "run-succeeded", requestId, revision: projectRevision, artifacts });
  } catch (error) {
    dispatch({ type: "run-failed", requestId, revision: projectRevision, error: message(error) });
  }
}

async function runSingle(request) {
  if (state.compile.status !== "compiled" || state.run.status === "running") return;
  const requestId = crypto.randomUUID();
  const projectRevision = state.projectRevision;
  dispatch({ type: "run-started", requestId, revision: projectRevision });
  try {
    const result = await runtime.run(request, renderProgress);
    dispatch({ type: "run-succeeded", requestId, revision: projectRevision, artifacts: result.artifacts });
  } catch (error) {
    dispatch({ type: "run-failed", requestId, revision: projectRevision, error: message(error) });
  }
}

function showFollowupError(label, error) {
  element("#run-error").textContent = `Posterior completed; ${label} unavailable: ${message(error)}`;
  element("#run-error").hidden = false;
}

function compiledBytes() {
  if (state.compile.status !== "compiled") throw new Error("Compile the model first");
  return state.compile.irBytes;
}

function documentBytes(text) {
  return new TextEncoder().encode(serializeDocument(parseDocument(text)));
}

function samplerSettings() {
  return {
    chains: integerValue("#chains"),
    num_warmup: integerValue("#warmup"),
    num_draws: integerValue("#draws"),
    seed: integerValue("#seed"),
    target_accept: numberValue("#target-accept"),
    max_treedepth: integerValue("#max-treedepth"),
  };
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
  element("#compile-status").textContent = state.compile.status === "compiling" ? "Compiling in an isolated worker…" : "";
  const compileError = element("#compile-error");
  compileError.hidden = state.compile.status !== "failed";
  compileError.textContent = state.compile.status === "failed" ? state.compile.error : "";
  element("#compile-button").disabled = state.compile.status === "compiling" || state.source.trim() === "";
  element("#share-button").disabled = state.source.trim() === "";
  const unavailable = state.compile.status !== "compiled" || state.run.status === "running";
  element("#sample-button").disabled = unavailable || observed.value.trim() === "";
  element("#prior-button").disabled = unavailable || design.value.trim() === "";
  element("#simulate-button").disabled = unavailable || design.value.trim() === "" || truth.value.trim() === "";
  element("#sample-simulated-button").disabled = unavailable ||
    !state.artifacts.some((artifact) => artifact.name === "simulated_data.json") || truth.value.trim() === "";

  const runError = element("#run-error");
  if (state.run.status === "failed") {
    runError.hidden = false;
    runError.textContent = state.run.error;
  } else if (!runError.textContent.startsWith("Posterior completed")) {
    runError.hidden = true;
    runError.textContent = "";
  }
  renderArtifacts(state.artifacts);
}

function renderArtifacts(artifacts) {
  for (const url of objectUrls) URL.revokeObjectURL(url);
  objectUrls.clear();
  const mapping = {
    "posterior.ndjson": "#artifact-posterior",
    "diagnostics.json": "#artifact-diagnostics",
    "prior_predictive.ndjson": "#artifact-prior",
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
  renderPlots(artifacts);
}

function renderPlots(artifacts) {
  const posterior = artifacts.find((artifact) => artifact.name === "posterior.ndjson");
  const diagnostics = artifacts.find((artifact) => artifact.name === "diagnostics.json");
  const plots = element("#plots");
  if (posterior === undefined || diagnostics === undefined) {
    plots.hidden = true;
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
    plots.hidden = false;
  } catch (error) {
    plots.hidden = true;
    element("#run-error").hidden = false;
    element("#run-error").textContent = `Artifacts downloaded, but plots could not render: ${message(error)}`;
  }
}

function integerValue(selector) { return Number.parseInt(element(selector).value, 10); }
function numberValue(selector) { return Number(element(selector).value); }
function element(selector) {
  const value = document.querySelector(selector);
  if (value === null) throw new Error(`Missing required element ${selector}`);
  return value;
}
function message(error) { return error instanceof Error ? error.message : String(error); }

render();
void initializeExamples();
void initializeSharedProject();
