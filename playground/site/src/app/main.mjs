import { basicSetup, EditorView, python } from "../../vendor/codemirror/codemirror.mjs";
import { compile } from "../compile/index.mjs";
import {
  renderDensityOverlay,
  renderPriorPosteriorOverlay,
} from "../critique/render.mjs";
import { bind, importCsv, importJson, requiredInputs, standardize } from "../data/index.mjs";
import {
  readDashboardData,
  renderEssRhat,
  renderPrecis,
  renderTrank,
} from "../dashboard/index.mjs";
import { WorkerEngine } from "../engine/executor.mjs";
import {
  diagnose,
  mergeChainFits,
  posteriorPredictive,
  priorPredictive,
  sample,
} from "../engine/verbs.mjs";

const UTF8 = new TextDecoder();
const editorHost = element('[data-testid="model-editor"]');
const hashChip = element("#ir-hash-chip");
const compileError = element("#compile-error");
const mappingBody = element("#mapping-table tbody");
const runButton = element("#run-button");
const progress = element("#progress");
const plotMode = element("#plot-mode");

let compileTimer;
let compileGeneration = 0;
let compiled = null;
let dataSource = null;
let boundDocument = null;
let mappingComplete = false;
let running = false;
let lastRun = null;
let dashboardData = null;
let fitDownload = null;
let diagnosticsDownload = null;

const editor = new EditorView({
  doc: "",
  extensions: [
    basicSetup,
    python(),
    EditorView.contentAttributes.of({ "aria-label": "Model source" }),
    EditorView.updateListener.of((update) => {
      if (update.docChanged) scheduleCompile();
    }),
  ],
  parent: editorHost,
});

window.__playground = {
  setSource(source) {
    editor.dispatch({
      changes: { from: 0, to: editor.state.doc.length, insert: source },
    });
  },
  getSource() {
    return editor.state.doc.toString();
  },
  state() {
    return {
      irHash: compiled?.irHash ?? "",
      mappingComplete,
      running,
      lastRun: lastRun === null ? null : { ...lastRun },
    };
  },
};

element("#json-load").addEventListener("click", loadJson);
element("#csv-input").addEventListener("change", loadCsv);
element("#sampler-settings").addEventListener("submit", (event) => {
  event.preventDefault();
  void runStudy();
});
plotMode.addEventListener("change", renderChainPlot);
element("#download-fit").addEventListener("click", () => {
  if (fitDownload !== null) downloadBlob(fitDownload, "fit.ndjson");
});
element("#download-diagnostics").addEventListener("click", () => {
  if (diagnosticsDownload !== null) {
    downloadBlob(diagnosticsDownload, "diagnostics.json");
  }
});
for (const button of document.querySelectorAll(".download-svg")) {
  button.addEventListener("click", () => downloadPlot(button.dataset.plot));
}

function scheduleCompile() {
  window.clearTimeout(compileTimer);
  compileGeneration += 1;
  compiled = null;
  mappingComplete = false;
  boundDocument = null;
  hashChip.textContent = "";
  compileError.hidden = true;
  renderMapping([]);
  updateRunButton();
  const generation = compileGeneration;
  compileTimer = window.setTimeout(() => {
    void compileSource(generation);
  }, 400);
}

async function compileSource(generation) {
  const result = await compile(editor.state.doc.toString());
  if (generation !== compileGeneration) return;
  if (!result.ok) {
    compileError.textContent = `${result.message}\n${result.traceback}`;
    compileError.hidden = false;
    hashChip.textContent = "";
    updateRunButton();
    return;
  }
  const ir = JSON.parse(UTF8.decode(result.irBytes));
  compiled = { ...result, ir };
  hashChip.textContent = result.irHash;
  compileError.textContent = "";
  compileError.hidden = true;
  rebindData();
}

function loadJson() {
  try {
    const documentValue = importJson(element("#json-input").value);
    dataSource = {
      kind: "json",
      original: structuredClone(documentValue),
      standardized: new Set(),
    };
    rebindData();
  } catch (error) {
    showApplicationError(error);
  }
}

async function loadCsv(event) {
  const file = event.target.files?.[0];
  if (file === undefined) return;
  try {
    const imported = importCsv(await file.text());
    dataSource = {
      kind: "csv",
      original: imported.columns,
      standardized: new Set(),
    };
    rebindData();
  } catch (error) {
    showApplicationError(error);
  }
}

function rebindData() {
  if (compiled === null || dataSource === null) {
    boundDocument = null;
    mappingComplete = false;
    renderMapping([]);
    updateRunButton();
    return;
  }
  const inputs = requiredInputs(compiled.ir);
  if (dataSource.kind === "json") bindJson(inputs);
  else bindCsv(inputs);
  updateRunButton();
}

function bindJson(inputs) {
  const documentValue = structuredClone(dataSource.original);
  for (const name of dataSource.standardized) {
    const variable = documentValue.variables[name];
    if (variable === undefined) continue;
    const transformed = standardize({
      name,
      dtype: variable.dtype,
      values: variable.values,
    });
    variable.dtype = transformed.dtype;
    variable.values = transformed.values;
  }
  const mapping = inputs.map((input) => ({
    input: input.name,
    source: input.name in documentValue.variables ? "document" : null,
    status: input.name in documentValue.variables ? "bound" : "missing",
  }));
  boundDocument = documentValue;
  mappingComplete = mapping.every((row) => row.status === "bound");
  renderMapping(mapping, numericJsonVariables(documentValue));
}

function bindCsv(inputs) {
  const columns = dataSource.original.map((column) =>
    dataSource.standardized.has(column.name) ? standardize(column) : column,
  );
  const result = bind(inputs, columns);
  boundDocument = result.document;
  mappingComplete = result.complete;
  renderMapping(
    result.mapping,
    columns.filter((column) => column.dtype !== "string" && column.values.length > 1),
  );
}

function numericJsonVariables(documentValue) {
  return Object.entries(documentValue.variables)
    .filter(([, variable]) => variable.dtype !== "string" && variable.values.length > 1)
    .map(([name]) => ({ name }));
}

function renderMapping(mapping, numericValues = []) {
  const toggles = new Set(numericValues.map((value) => value.name));
  mappingBody.replaceChildren();
  for (const row of mapping) {
    const tableRow = document.createElement("tr");
    tableRow.dataset.input = row.input;
    tableRow.dataset.status = row.status;
    tableRow.append(
      tableCell(row.input),
      tableCell(row.source ?? "—"),
      tableCell(row.status, `mapping-${row.status}`),
      transformCell(row.input, toggles.has(row.input)),
    );
    mappingBody.append(tableRow);
  }
}

function transformCell(name, available) {
  const cell = document.createElement("td");
  if (!available || dataSource === null) return cell;
  const input = document.createElement("input");
  input.type = "checkbox";
  input.className = "standardize-toggle";
  input.dataset.column = name;
  input.id = `standardize-${name}`;
  input.checked = dataSource.standardized.has(name);
  input.addEventListener("change", () => {
    if (input.checked) dataSource.standardized.add(name);
    else dataSource.standardized.delete(name);
    rebindData();
  });
  const label = document.createElement("label");
  label.className = "standardize-label";
  label.htmlFor = input.id;
  label.append(input, "standardize");
  cell.append(label);
  return cell;
}

function tableCell(text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  cell.className = className;
  return cell;
}

function updateRunButton() {
  runButton.disabled = running || compiled === null || !mappingComplete;
}

async function runStudy() {
  if (compiled === null || boundDocument === null || !mappingComplete || running) return;
  const settings = samplerSettings();
  const executor = new WorkerEngine();
  const counts = Array.from({ length: settings.chains }, () => ({ draws: 0, divergences: 0 }));
  running = true;
  lastRun = {
    seed: settings.seed,
    chains: settings.chains,
    drawsStreamed: 0,
    divergences: 0,
  };
  resetResults();
  renderProgress(counts);
  updateRunButton();

  try {
    const sampled = requireResult(
      await sample({
        model: compiled.ir,
        data: boundDocument,
        settings: {
          num_warmup: settings.numWarmup,
          num_draws: settings.numDraws,
          max_treedepth: settings.maxTreedepth,
          target_accept: settings.targetAccept,
        },
        seed: settings.seed,
        chains: settings.chains,
        executor,
        onDrawBatch(batch) {
          const count = counts[batch.chainId];
          if (count === undefined) return;
          count.draws += batch.draws.length;
          count.divergences += batch.draws.filter((draw) => draw.diverging === true).length;
          lastRun.drawsStreamed += batch.draws.length;
          lastRun.divergences += batch.draws.filter((draw) => draw.diverging === true).length;
          renderProgress(counts);
        },
      }),
    );
    const fitTexts = sampled.map((output) => UTF8.decode(output.rawBytes));
    const mergedFit = mergeChainFits(fitTexts);
    const diagnosed = requireResult(await diagnose({ fits: fitTexts, executor }))[0];
    const predictive = requireResult(
      await posteriorPredictive({
        model: compiled.ir,
        data: boundDocument,
        fit: mergedFit,
        seed: settings.seed + 1,
        executor,
      }),
    )[0];
    const prior = requireResult(
      await priorPredictive({
        model: compiled.ir,
        data: predictiveInputs(compiled.ir, boundDocument),
        settings: { num_draws: 200 },
        seed: settings.seed + 2,
        executor,
      }),
    )[0];
    renderResults(fitTexts, diagnosed, predictive, prior);
  } catch (error) {
    showApplicationError(error);
  } finally {
    running = false;
    updateRunButton();
  }
}

function samplerSettings() {
  return {
    chains: integerValue("#chains"),
    numWarmup: integerValue("#num-warmup"),
    numDraws: integerValue("#num-draws"),
    seed: integerValue("#seed"),
    targetAccept: numberValue("#target-accept"),
    maxTreedepth: integerValue("#max-treedepth"),
  };
}

function renderProgress(counts) {
  progress.replaceChildren();
  for (const [chain, count] of counts.entries()) {
    const line = document.createElement("div");
    line.className = "chain-progress";
    line.dataset.chain = String(chain);
    line.textContent = `chain ${String(chain + 1)} · ${String(count.draws)} draws · ${String(count.divergences)} divergent`;
    progress.append(line);
  }
}

function renderResults(fitTexts, diagnosed, predictive, prior) {
  dashboardData = readDashboardData({ fits: fitTexts, diagnose: diagnosed.rawBytes });
  renderChainPlot();
  element("#plot-esshat").innerHTML = renderEssRhat(dashboardData);
  element("#plot-precis").innerHTML = renderPrecis(dashboardData);
  element("#plot-ppc").innerHTML = renderDensityOverlay(
    predictivePlotData(compiled.ir, boundDocument, predictive.rawBytes),
  );
  element("#plot-overlay").innerHTML = renderPriorPosteriorOverlay(
    overlayPlotData(dashboardData, prior.rawBytes),
  );

  fitDownload = new Blob([fitTexts.join("\n")], { type: "application/x-ndjson" });
  diagnosticsDownload = new Blob([diagnosed.rawBytes], { type: "application/json" });
  element("#download-fit").disabled = false;
  element("#download-diagnostics").disabled = false;
  for (const button of document.querySelectorAll(".download-svg")) button.disabled = false;
}

function renderChainPlot() {
  if (dashboardData === null) return;
  element("#plot-trank").innerHTML = renderTrank(dashboardData, plotMode.value);
}

function predictiveInputs(ir, documentValue) {
  const observed = new Set(ir.model.observed_nodes.map((node) => node.name));
  return {
    ...documentValue,
    variables: Object.fromEntries(
      Object.entries(documentValue.variables).filter(([name]) => !observed.has(name)),
    ),
  };
}

function predictivePlotData(ir, documentValue, bytes) {
  const names = ir.model.observed_nodes.map((node) => node.name);
  const observed = [];
  const labels = [];
  for (const name of names) {
    const values = documentValue.variables[name]?.values ?? [];
    for (const [index, value] of values.entries()) {
      observed.push(value);
      labels.push(`${name}[${String(index)}]`);
    }
  }
  const replicates = ndjsonDraws(bytes).map((draw) =>
    names.flatMap((name) => arrayValue(draw.values?.[name])),
  );
  return { observed, labels, replicates };
}

function overlayPlotData(data, bytes) {
  const priorDraws = ndjsonDraws(bytes);
  return data.parameters.map((parameter) => ({
    name: parameter.label,
    prior: priorDraws.map((draw) =>
      coordinateValue(draw.values?.[parameter.name], parameter.coordinate),
    ),
    posterior: parameter.chains.flat(),
  }));
}

function ndjsonDraws(bytes) {
  const lines = UTF8.decode(bytes).trimEnd().split("\n");
  return lines.slice(1, -1).map((line) => JSON.parse(line));
}

function coordinateValue(value, coordinate) {
  if (coordinate.length === 0) return value;
  return coordinate.reduce((current, index) => current[index], value);
}

function arrayValue(value) {
  return Array.isArray(value) ? value : [value];
}

function requireResult(result) {
  if (!result.ok) throw new Error(`${result.error.error}: ${result.error.message}`);
  return result.outputs;
}

function resetResults() {
  dashboardData = null;
  fitDownload = null;
  diagnosticsDownload = null;
  for (const id of ["trank", "esshat", "precis", "ppc", "overlay"]) {
    element(`#plot-${id}`).replaceChildren();
  }
  element("#download-fit").disabled = true;
  element("#download-diagnostics").disabled = true;
  for (const button of document.querySelectorAll(".download-svg")) button.disabled = true;
}

function showApplicationError(error) {
  compileError.textContent = error instanceof Error ? error.message : String(error);
  compileError.hidden = false;
}

function downloadPlot(name) {
  const svg = element(`#plot-${name} svg`);
  downloadBlob(new Blob([svg.outerHTML], { type: "image/svg+xml" }), `${name}.svg`);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function integerValue(selector) {
  return Number.parseInt(element(selector).value, 10);
}

function numberValue(selector) {
  return Number(element(selector).value);
}

function element(selector) {
  const found = document.querySelector(selector);
  if (found === null) throw new Error(`Missing application element ${selector}`);
  return found;
}
