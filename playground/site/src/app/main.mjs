import { compile } from "../compile/index.mjs";
import { parseDocument, serializeDocument } from "../data/documents.mjs";
import { readDashboardData, renderEssRhat, renderPrecis, renderTrank } from "../dashboard/index.mjs";
import { BrowserRuntime } from "../runtime/browser-runtime.mjs";
import { initialState, reduce } from "./state.mjs";

const runtime = new BrowserRuntime();
let state = initialState();
let revision = 0;
const objectUrls = new Set();

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
  if (state.compile.status !== "compiled") return;
  const requestId = crypto.randomUUID();
  const projectRevision = state.projectRevision;
  element("#progress").replaceChildren();
  dispatch({ type: "run-started", requestId, revision: projectRevision });
  try {
    const dataBytes = new TextEncoder().encode(serializeDocument(parseDocument(observed.value)));
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
      element("#run-error").textContent = `Posterior completed; diagnostics unavailable: ${message(error)}`;
      element("#run-error").hidden = false;
    }
    dispatch({ type: "run-succeeded", requestId, revision: projectRevision, artifacts });
  } catch (error) {
    dispatch({ type: "run-failed", requestId, revision: projectRevision, error: message(error) });
  }
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
  element("#sample-button").disabled = state.compile.status !== "compiled" || state.run.status === "running" || observed.value.trim() === "";

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
