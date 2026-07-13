import { basicSetup, EditorView, python } from "../../vendor/codemirror/codemirror.mjs";
import { compile } from "../compile/index.mjs";
import {
  renderDensityOverlay,
  renderPriorPosteriorOverlay,
  renderPriorPredictiveDensity,
} from "../critique/render.mjs";
import {
  defaultTruth,
  designDefaults,
  designDocument,
  truthDocument,
} from "./design.mjs";
import { exampleAssetUrl } from "./examples.mjs";
import { decodeProject, encodeProject, FRAGMENT_WARN_LENGTH } from "./share.mjs";
import { subsampleReplicates } from "./subsample.mjs";
import { bind, importCsv, importJson, requiredInputs, standardize } from "../data/index.mjs";
import {
  quantile,
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
  simulate,
} from "../engine/verbs.mjs";

const UTF8 = new TextDecoder();
const editorHost = element('[data-testid="model-editor"]');
const hashChip = element("#ir-hash-chip");
const compileError = element("#compile-error");
const runError = element("#run-error");
const mappingBody = element("#mapping-table tbody");
const runButton = element("#run-button");
const progress = element("#progress");
const plotMode = element("#plot-mode");
const observedPanel = element("#observed-panel");
const designPanel = element("#design-panel");
const designRunControls = element("#design-run-controls");
const designBody = element("#design-values tbody");
const truthBody = element("#truth-values tbody");
const priorPredictiveButton = element("#run-prior-predictive");
const simulateButton = element("#run-simulate");
const simulateReason = element("#simulate-reason");
const shareButton = element("#share-button");
const shareUrl = element("#share-url");
const shareWarning = element("#share-warning");
const shareInterstitial = element("#share-interstitial");
const examplesMenu = element("#examples-menu");

let compileTimer;
let compileGeneration = 0;
let compiled = null;
let dataSource = null;
let columnAssignments = {};
let boundDocument = null;
let mappingComplete = false;
let running = false;
let lastRun = null;
let dashboardData = null;
let fitDownload = null;
let diagnosticsDownload = null;
let dataMode = "observed";
let simulated = false;
let simulatedDocument = null;
let simulatedTruth = null;
let recovery = null;
let resolvedTruthSizes = {};
let truthSizeError = null;
let partiallyObserved = false;
let pendingSharedForms = null;
let sharedProject = null;
let examples = new Map();
let preparedShare = null;
let sharePreparation = 0;

const editor = new EditorView({
  doc: "",
  extensions: [
    basicSetup,
    python(),
    EditorView.contentAttributes.of({ "aria-label": "Model source" }),
    EditorView.updateListener.of((update) => {
      if (!update.docChanged) return;
      updateShareButton();
      void prepareSharePayload();
      scheduleCompile();
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
      dataMode,
      simulated,
      recovery: recovery === null ? null : structuredClone(recovery),
    };
  },
};

element("#json-load").addEventListener("click", loadJson);
element("#csv-input").addEventListener("change", loadCsv);
element("#data-mode-observed").addEventListener("change", changeDataMode);
element("#data-mode-design").addEventListener("change", changeDataMode);
shareButton.addEventListener("pointerdown", () => void prepareSharePayload());
shareButton.addEventListener("click", () => void shareProject());
element("#share-load").addEventListener("click", loadSharedProject);
examplesMenu.addEventListener("change", () => void loadExample());
element("#sampler-settings").addEventListener("input", () => void prepareSharePayload());
designBody.addEventListener("input", designInputsChanged);
truthBody.addEventListener("input", designInputsChanged);
priorPredictiveButton.addEventListener("click", () => void runPriorPredictive());
simulateButton.addEventListener("click", () => void runSimulation());
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

void initializeExamples();
void initializeSharedProject();

function updateShareButton() {
  shareButton.disabled = editor.state.doc.length === 0;
}

function currentShareProject() {
  const settings = samplerSettings();
  return {
    v: 1,
    source: editor.state.doc.toString(),
    dataMode,
    design: currentDesign(),
    truth: currentTruth(),
    sampler: {
      chains: settings.chains,
      num_warmup: settings.numWarmup,
      num_draws: settings.numDraws,
      seed: settings.seed,
      target_accept: settings.targetAccept,
      max_treedepth: settings.maxTreedepth,
    },
  };
}

async function prepareSharePayload() {
  if (editor.state.doc.length === 0) {
    preparedShare = null;
    return;
  }
  const project = currentShareProject();
  const json = JSON.stringify(project);
  const generation = ++sharePreparation;
  try {
    const payload = await encodeProject(project);
    if (generation === sharePreparation) preparedShare = { json, payload };
  } catch (error) {
    showApplicationError(error);
  }
}

async function shareProject() {
  if (editor.state.doc.length === 0) return;
  try {
    const project = currentShareProject();
    const json = JSON.stringify(project);
    const payload = preparedShare?.json === json ? preparedShare.payload : await encodeProject(project);
    window.location.hash = `project=${payload}`;
    shareUrl.value = window.location.href;
    element(".share-output").hidden = false;
    shareWarning.hidden = payload.length <= FRAGMENT_WARN_LENGTH;
  } catch (error) {
    showApplicationError(error);
  }
}

async function initializeSharedProject() {
  if (!window.location.hash.startsWith("#project=")) return;
  const payload = window.location.hash.slice("#project=".length);
  try {
    sharedProject = await decodeProject(payload);
    element("#share-source").textContent = sharedProject.source;
    shareInterstitial.hidden = false;
  } catch (error) {
    showApplicationError(error);
  }
}

function loadSharedProject() {
  if (sharedProject === null) return;
  applySamplerSettings(sharedProject.sampler);
  setDataMode(sharedProject.dataMode);
  pendingSharedForms = {
    design: sharedProject.design ?? {},
    truth: sharedProject.truth ?? {},
  };
  shareInterstitial.hidden = true;
  editor.dispatch({
    changes: {
      from: 0,
      to: editor.state.doc.length,
      insert: sharedProject.source,
    },
  });
  sharedProject = null;
}

function applySamplerSettings(settings) {
  const values = {
    "#chains": settings.chains,
    "#num-warmup": settings.num_warmup,
    "#num-draws": settings.num_draws,
    "#seed": settings.seed,
    "#target-accept": settings.target_accept,
    "#max-treedepth": settings.max_treedepth,
  };
  for (const [selector, value] of Object.entries(values)) {
    if (value !== undefined) element(selector).value = String(value);
  }
}

function applySharedForms(forms) {
  for (const row of designBody.querySelectorAll("tr[data-design-name]")) {
    const values = forms.design[row.dataset.designName];
    if (values === undefined) continue;
    const fields = row.dataset.designKind === "scalar" ? ["value"] : ["low", "high", "n"];
    for (const field of fields) {
      if (values[field] !== undefined) {
        row.querySelector(`[data-design-field="${field}"]`).value = String(values[field]);
      }
    }
  }
  for (const row of truthBody.querySelectorAll("tr[data-truth-name]")) {
    const value = forms.truth[row.dataset.truthName];
    if (value !== undefined) row.querySelector("input[type=number]").value = String(value);
  }
}

async function initializeExamples() {
  try {
    const response = await fetch(exampleAssetUrl("EXAMPLES.json"));
    if (!response.ok) throw new Error(`Examples manifest request failed: HTTP ${response.status}`);
    const entries = await response.json();
    examples = new Map(entries.map((entry) => [entry.id, entry]));
    for (const entry of entries) {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.title;
      examplesMenu.append(option);
    }
  } catch (error) {
    showApplicationError(error);
  }
}

async function loadExample() {
  const entry = examples.get(examplesMenu.value);
  if (entry === undefined) return;
  try {
    const sourceResponse = await fetch(exampleAssetUrl(entry.source_path));
    if (!sourceResponse.ok) {
      throw new Error(`Example source request failed: HTTP ${sourceResponse.status}`);
    }
    const source = await sourceResponse.text();
    editor.dispatch({
      changes: { from: 0, to: editor.state.doc.length, insert: source },
    });
    if (entry.data_path !== undefined) {
      const dataResponse = await fetch(exampleAssetUrl(entry.data_path));
      if (!dataResponse.ok) {
        throw new Error(`Example data request failed: HTTP ${dataResponse.status}`);
      }
      element("#json-input").value = await dataResponse.text();
      loadJson();
    }
  } catch (error) {
    showApplicationError(error);
  }
}

function scheduleCompile() {
  window.clearTimeout(compileTimer);
  compileGeneration += 1;
  compiled = null;
  mappingComplete = false;
  boundDocument = null;
  simulated = false;
  simulatedDocument = null;
  simulatedTruth = null;
  recovery = null;
  resolvedTruthSizes = {};
  truthSizeError = null;
  partiallyObserved = false;
  runButton.textContent = "Run sampler";
  resetResults();
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
  partiallyObserved = containsNode(ir.model, "VectorScatterOp");
  hashChip.textContent = result.irHash;
  renderDesignTables(ir);
  if (pendingSharedForms !== null) {
    applySharedForms(pendingSharedForms);
    pendingSharedForms = null;
  }
  refreshTruthSizes();
  void prepareSharePayload();
  compileError.textContent = "";
  compileError.hidden = true;
  rebindData();
}

function changeDataMode(event) {
  if (!event.target.checked) return;
  setDataMode(event.target.value);
}

function setDataMode(mode) {
  dataMode = mode === "design" ? "design" : "observed";
  element(`#data-mode-${dataMode}`).checked = true;
  observedPanel.hidden = dataMode !== "observed";
  designPanel.hidden = dataMode !== "design";
  designRunControls.hidden = dataMode !== "design";
  rebindData();
  void prepareSharePayload();
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
    columnAssignments = {};
    rebindData();
  } catch (error) {
    showApplicationError(error);
  }
}

function rebindData() {
  if (compiled === null) {
    boundDocument = null;
    mappingComplete = false;
    renderMapping([]);
    updateRunButton();
    return;
  }
  if (dataMode === "design") {
    if (simulatedDocument === null) {
      boundDocument = null;
      mappingComplete = false;
      renderMapping([]);
    } else {
      bindSimulated(requiredInputs(compiled.ir));
    }
    updateRunButton();
    return;
  }
  if (dataSource === null) {
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

function bindSimulated(inputs) {
  const variables = simulatedDocument.variables;
  const missing = inputs.filter(
    (input) => input.synthetic !== true && variables[input.name] === undefined,
  );
  if (missing.length > 0) {
    throw new Error(
      `Simulated data is missing required inputs: ${missing.map((input) => input.name).join(", ")}`,
    );
  }
  boundDocument = simulatedDocument;
  mappingComplete = true;
  renderMapping(
    inputs.map((input) => ({ input: input.name, source: "simulated", status: "bound" })),
  );
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

  const lengthGroups = new Map();
  for (const input of inputs) {
    if (input.kind !== "vector") continue;
    const variable = documentValue.variables[input.name];
    if (variable === undefined) continue;
    const dimensions = input.dims.length === 0
      ? [["n", variable.shape[0] ?? variable.values.length]]
      : input.dims.map((dim, index) => [dim, variable.shape[index]]);
    for (const [group, length] of dimensions) {
      if (length === undefined) {
        throw new Error(
          `bound vector ${input.name} has no shape entry for dimension ${group}`,
        );
      }
      const existing = lengthGroups.get(group);
      if (existing !== undefined && length !== existing.length) {
        throw new Error(
          `bound vector length mismatch for group ${group}: ` +
            `${input.name} has ${String(length)} rows, ` +
            `${existing.input} has ${String(existing.length)}`,
        );
      }
      lengthGroups.set(group, { input: input.name, length });
    }
  }

  const mapping = inputs.map((input) => {
    if (input.name in documentValue.variables) {
      return { input: input.name, source: "document", status: "bound" };
    }
    const lengthGroup = input.kind === "scalar"
      ? lengthGroups.get(input.synthetic === true ? "n" : input.name)
      : undefined;
    if (lengthGroup !== undefined) {
      // Synthetic lengths are UI bookkeeping only: the IR declares no such
      // variable, and the engine rejects undeclared data. Only dim-named
      // scalars the model actually declares are materialized.
      if (input.synthetic !== true) {
        documentValue.variables[input.name] = {
          dtype: "int64",
          shape: [],
          values: [lengthGroup.length],
        };
      }
      return { input: input.name, source: "auto:length", status: "bound" };
    }
    return { input: input.name, source: null, status: "missing" };
  });
  boundDocument = documentValue;
  mappingComplete = mapping.every((row) => row.status === "bound");
  renderMapping(mapping, numericJsonVariables(documentValue));
}

function bindCsv(inputs) {
  const columns = dataSource.original.map((column) =>
    dataSource.standardized.has(column.name) ? standardize(column) : column,
  );
  const result = bind(inputs, columns, columnAssignments);
  boundDocument = result.document;
  mappingComplete = result.complete;
  renderMapping(
    result.mapping,
    columns.filter((column) => column.dtype !== "string"),
    inputs,
  );
}

function numericJsonVariables(documentValue) {
  return Object.entries(documentValue.variables)
    .filter(([, variable]) => variable.dtype !== "string" && variable.values.length > 1)
    .map(([name]) => ({ name }));
}

function renderDesignTables(ir) {
  designBody.replaceChildren();
  const defaultsByName = designDefaults(ir);
  for (const [name, defaults] of Object.entries(defaultsByName)) {
    const row = document.createElement("tr");
    row.dataset.designName = name;
    if (Object.hasOwn(defaults, "value")) {
      row.dataset.designKind = "scalar";
      row.dataset.designCountLike = String(defaults.countLike !== false);
      row.append(
        tableCell(name),
        tableCell("—"),
        tableCell("—"),
        numberInputCell(
          `design-${name}-value`,
          `Scalar value for ${name}`,
          "value",
          defaults.value,
          defaults.countLike !== false,
        ),
      );
    } else {
      row.dataset.designKind = "vector";
      row.append(
        tableCell(name),
        numberInputCell(`design-${name}-low`, `Low for ${name}`, "low", defaults.low),
        numberInputCell(`design-${name}-high`, `High for ${name}`, "high", defaults.high),
        numberInputCell(`design-${name}-n`, `Rows for ${name}`, "n", defaults.n, true),
      );
    }
    designBody.append(row);
  }
  const designedNames = new Set(Object.keys(defaultsByName));
  const inputs = requiredInputs(ir);
  for (const input of inputs) {
    if (input.kind !== "scalar" || designedNames.has(input.name)) continue;
    const derived = input.synthetic === true ||
      inputs.some(
        (candidate) => candidate.kind === "vector" && candidate.dims.includes(input.name),
      );
    if (!derived) continue;
    const row = document.createElement("tr");
    row.dataset.designAutoName = input.name;
    const automaticCell = document.createElement("td");
    const automatic = document.createElement("input");
    automatic.type = "text";
    automatic.value = "auto";
    automatic.readOnly = true;
    automatic.setAttribute("aria-label", `Automatic value for ${input.name}`);
    automaticCell.append(automatic);
    row.append(tableCell(input.name), tableCell("auto"), tableCell("auto"), automaticCell);
    designBody.append(row);
  }

  truthBody.replaceChildren();
  const paramsByName = new Map(ir.model.params.map((parameter) => [parameter.name, parameter]));
  for (const [name, value] of Object.entries(defaultTruth(ir))) {
    const row = document.createElement("tr");
    row.dataset.truthName = name;
    const parameter = paramsByName.get(name);
    const sizeName = parameter?.value.size?.name;
    const ordered = parameter?.value.constraint?.node === "Ordered";
    if (sizeName !== undefined) row.dataset.truthSize = sizeName;
    if (ordered) row.dataset.truthOrdered = "true";
    const valueCell = numberInputCell(
      `truth-${name}`,
      `Truth for ${name}`,
      undefined,
      value,
    );
    valueCell.querySelector("input").dataset.truthValue = "";
    const truthLabel = ordered
      ? `${name} (ordered around v)`
      : sizeName === undefined
        ? name
        : `${name} × ${sizeName}`;
    row.append(tableCell(truthLabel), valueCell);
    truthBody.append(row);
  }
}

function numberInputCell(id, labelText, field, value, integer = false) {
  const cell = document.createElement("td");
  const label = document.createElement("label");
  label.className = "visually-hidden";
  label.htmlFor = id;
  label.textContent = labelText;
  const input = document.createElement("input");
  input.id = id;
  input.type = "number";
  input.value = String(value);
  input.step = integer ? "1" : "any";
  if (integer) input.min = "1";
  if (field !== undefined) input.dataset.designField = field;
  cell.append(label, input);
  return cell;
}

function currentDesign() {
  return Object.fromEntries(
    [...designBody.querySelectorAll("tr[data-design-name]")].map((row) => {
      if (row.dataset.designKind === "scalar") {
        return [
          row.dataset.designName,
          {
            value: Number(row.querySelector('[data-design-field="value"]').value),
            countLike: row.dataset.designCountLike !== "false",
          },
        ];
      }
      return [
        row.dataset.designName,
        {
          low: Number(row.querySelector('[data-design-field="low"]').value),
          high: Number(row.querySelector('[data-design-field="high"]').value),
          n: Number(row.querySelector('[data-design-field="n"]').value),
        },
      ];
    }),
  );
}

function currentTruth() {
  return Object.fromEntries(
    [...truthBody.querySelectorAll("tr[data-truth-name]")].map((row) => [
      row.dataset.truthName,
      Number(row.querySelector("input[type=number]").value),
    ]),
  );
}

function designInputsChanged() {
  refreshTruthSizes();
  void prepareSharePayload();
  simulated = false;
  simulatedDocument = null;
  simulatedTruth = null;
  recovery = null;
  boundDocument = null;
  mappingComplete = false;
  runButton.textContent = "Run sampler";
  resetResults();
  rebindData();
}

function refreshTruthSizes() {
  resolvedTruthSizes = {};
  truthSizeError = null;
  if (compiled === null) return;
  try {
    const documentValue = boundDesignDocument();
    for (const parameter of compiled.ir.model.params) {
      const size = parameter.value.size;
      if (size === null || size === undefined) continue;
      if (size.node !== "DataRef") {
        throw new Error(`cannot resolve truth size for ${parameter.name}`);
      }
      const value = documentValue.variables[size.name]?.values?.[0];
      if (!Number.isInteger(value) || value < 1) {
        throw new Error(
          `cannot resolve truth size ${size.name} for ${parameter.name} from design values`,
        );
      }
      resolvedTruthSizes[parameter.name] = parameter.value.constraint?.node === "Ordered"
        ? { size: value, ordered: true }
        : value;
    }
  } catch (error) {
    truthSizeError = error instanceof Error ? error.message : String(error);
  }
}

function boundDesignDocument() {
  const generated = designDocument(currentDesign());
  for (const name of indexDataNames(compiled.ir.model)) {
    const variable = generated.variables[name];
    if (variable === undefined || variable.shape.length === 0) continue;
    variable.dtype = "int64";
    variable.values = Array(variable.values.length).fill(0);
  }
  const columns = Object.entries(generated.variables)
    .filter(([, variable]) => variable.shape.length > 0)
    .map(([name, variable]) => ({
      name,
      dtype: variable.dtype,
      values: variable.values,
    }));
  const documentValue = bind(requiredInputs(compiled.ir), columns).document;
  for (const [name, variable] of Object.entries(generated.variables)) {
    if (variable.shape.length === 0) documentValue.variables[name] = variable;
  }
  return documentValue;
}

function indexDataNames(value, names = new Set()) {
  if (Array.isArray(value)) {
    for (const entry of value) indexDataNames(entry, names);
  } else if (value !== null && typeof value === "object") {
    if (value.node === "ScalarIndex" && value.expr?.node === "DataRef") {
      names.add(value.expr.name);
    }
    for (const entry of Object.values(value)) indexDataNames(entry, names);
  }
  return names;
}

function containsNode(value, node) {
  if (Array.isArray(value)) return value.some((entry) => containsNode(entry, node));
  if (value === null || typeof value !== "object") return false;
  if (value.node === node) return true;
  return Object.values(value).some((entry) => containsNode(entry, node));
}

function simulatedDataDocument(bytes) {
  let parsed;
  try {
    parsed = JSON.parse(UTF8.decode(bytes));
  } catch {
    throw new Error("Simulate output is not valid JSON");
  }
  let candidate;
  if (jsonObject(parsed) && jsonObject(parsed.variables)) {
    candidate = {
      format: parsed.format ?? "bayescycle.data.json.v1",
      variables: parsed.variables,
    };
  } else if (
    jsonObject(parsed) &&
    Object.values(parsed).every((value) => dataVariable(value))
  ) {
    candidate = { format: "bayescycle.data.json.v1", variables: parsed };
  } else {
    throw new Error(
      "Simulate output must be a data-document envelope or a variables object",
    );
  }
  try {
    return importJson(JSON.stringify(candidate));
  } catch (error) {
    throw new Error(
      `Simulate output is not a valid bayescycle data document: ${error.message}`,
    );
  }
}

function completeDerivedScalars(documentValue, inputs) {
  const completed = structuredClone(documentValue);
  for (const input of inputs.filter((candidate) => candidate.kind === "vector")) {
    const variable = completed.variables[input.name];
    if (variable === undefined) continue;
    for (const dimension of input.dims) {
      completed.variables[dimension] ??= {
        dtype: "int64",
        shape: [],
        values: [variable.values.length],
      };
    }
  }
  return completed;
}

function priorPredictiveReplicates(ir, bytes) {
  const names = ir.model.observed_nodes.map((node) => node.name);
  return ndjsonDraws(bytes).map((draw) =>
    names.flatMap((name) => arrayValue(draw.values?.[name])),
  );
}

function expandedRecoveryTruth(data, truth) {
  return Object.fromEntries(
    Object.entries(truth).flatMap(([name, value]) => {
      const parameters = data.parameters.filter(
        (candidate) => candidate.name === name || candidate.label === name,
      );
      if (parameters.length === 0) {
        throw new Error(`Posterior draws are missing truth parameter ${name}`);
      }
      return parameters.map((parameter) => [
        parameter.label,
        Array.isArray(value) ? coordinateValue(value, parameter.coordinate) : value,
      ]);
    }),
  );
}

function recoverySummary(data, truth) {
  return Object.fromEntries(
    Object.entries(truth).map(([name, value]) => {
      const parameter = data.parameters.find((candidate) => candidate.label === name);
      if (parameter === undefined) {
        throw new Error(`Posterior draws are missing truth parameter ${name}`);
      }
      const draws = parameter.chains.flat();
      const low = quantile(draws, 0.055);
      const high = quantile(draws, 0.945);
      return [name, { truth: value, low, high, inside: value >= low && value <= high }];
    }),
  );
}

function jsonObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function dataVariable(value) {
  return jsonObject(value) && Array.isArray(value.shape) && Array.isArray(value.values);
}

function renderMapping(mapping, numericValues = [], inputs = []) {
  const toggles = new Set(
    numericValues
      .filter((value) => value.values === undefined || value.values.length > 1)
      .map((value) => value.name),
  );
  const inputKinds = new Map(inputs.map((input) => [input.name, input.kind]));
  mappingBody.replaceChildren();
  for (const row of mapping) {
    const tableRow = document.createElement("tr");
    tableRow.dataset.input = row.input;
    tableRow.dataset.status = row.status;
    const sourceName = row.source?.startsWith("column:")
      ? row.source.slice("column:".length)
      : row.input;
    tableRow.append(
      tableCell(row.input),
      tableCell(row.source ?? "—"),
      tableCell(row.status, `mapping-${row.status}`),
      columnPickerCell(row.input, inputKinds.get(row.input), numericValues),
      transformCell(sourceName, toggles.has(sourceName)),
    );
    mappingBody.append(tableRow);
  }
}

function columnPickerCell(inputName, inputKind, columns) {
  const cell = document.createElement("td");
  if (dataSource?.kind !== "csv" || inputKind !== "vector") return cell;
  const select = document.createElement("select");
  select.className = "column-picker";
  select.dataset.input = inputName;
  select.setAttribute("aria-label", `Column for ${inputName}`);
  const automatic = document.createElement("option");
  automatic.value = "";
  automatic.textContent = "match by name…";
  select.append(automatic);
  for (const column of columns) {
    const option = document.createElement("option");
    option.value = column.name;
    option.textContent = column.name;
    select.append(option);
  }
  select.value = columnAssignments[inputName] ?? "";
  select.addEventListener("change", () => {
    if (select.value === "") delete columnAssignments[inputName];
    else columnAssignments[inputName] = select.value;
    rebindData();
  });
  cell.append(select);
  return cell;
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
  const designDisabled = running || compiled === null || dataMode !== "design";
  const partialReason = partiallyObserved
    ? "Design mode does not support partially observed models yet — bind observed data instead."
    : null;
  priorPredictiveButton.disabled = designDisabled || partialReason !== null;
  simulateButton.disabled = designDisabled || partialReason !== null || truthSizeError !== null;
  const reason = partialReason ?? truthSizeError;
  simulateReason.textContent = reason ?? "";
  simulateReason.hidden = reason === null || dataMode !== "design";
}

async function runPriorPredictive() {
  if (compiled === null || running || dataMode !== "design" || partiallyObserved) return;
  running = true;
  recovery = null;
  clearRunError();
  element("#plot-ppc").replaceChildren();
  updateRunButton();
  try {
    const executor = new WorkerEngine();
    const output = requireResult(
      await priorPredictive({
        model: compiled.ir,
        data: boundDesignDocument(),
        settings: { num_draws: 200 },
        seed: integerValue("#seed"),
        executor,
      }),
    )[0];
    element("#plot-ppc").innerHTML = renderPriorPredictiveDensity(
      subsampleReplicates(priorPredictiveReplicates(compiled.ir, output.rawBytes)),
    );
    element("#results").hidden = false;
  } catch (error) {
    showRunError(error);
  } finally {
    running = false;
    updateRunButton();
  }
}

async function runSimulation() {
  if (compiled === null || running || dataMode !== "design" || partiallyObserved) return;
  running = true;
  recovery = null;
  clearRunError();
  simulated = false;
  simulatedDocument = null;
  simulatedTruth = null;
  boundDocument = null;
  mappingComplete = false;
  renderMapping([]);
  resetResults();
  updateRunButton();
  try {
    const truth = currentTruth();
    const resolvedTruth = truthDocument(truth, resolvedTruthSizes);
    const executor = new WorkerEngine();
    const output = requireResult(
      await simulate({
        model: compiled.ir,
        data: boundDesignDocument(),
        truth: resolvedTruth,
        seed: integerValue("#seed"),
        executor,
      }),
    )[0];
    simulatedDocument = completeDerivedScalars(
      simulatedDataDocument(output.rawBytes),
      requiredInputs(compiled.ir),
    );
    simulatedTruth = Object.fromEntries(
      Object.entries(resolvedTruth.variables).map(([name, variable]) => [
        name,
        variable.shape.length === 0 ? variable.values[0] : variable.values,
      ]),
    );
    simulated = true;
    runButton.textContent = "Sample on simulated data";
    bindSimulated(requiredInputs(compiled.ir));
  } catch (error) {
    showRunError(error);
  } finally {
    running = false;
    updateRunButton();
  }
}

async function runStudy() {
  if (compiled === null || boundDocument === null || !mappingComplete || running) return;
  const settings = samplerSettings();
  const truthForRecovery =
    dataMode === "design" && simulated && simulatedTruth !== null
      ? structuredClone(simulatedTruth)
      : null;
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
  recovery = null;
  clearRunError();
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
    const predictive = partiallyObserved || compiled.ir.model.observed_nodes.length === 0
      ? null
      : requireResult(
          await posteriorPredictive({
            model: compiled.ir,
            data: boundDocument,
            fit: mergedFit,
            seed: settings.seed + 1,
            executor,
          }),
        )[0];
    const prior = partiallyObserved || compiled.ir.model.observed_nodes.length === 0
      ? null
      : requireResult(
          await priorPredictive({
            model: compiled.ir,
            data: predictiveInputs(compiled.ir, boundDocument),
            settings: { num_draws: 200 },
            seed: settings.seed + 2,
            executor,
          }),
        )[0];
    const downloadableFit = fitTexts.length === 1 ? fitTexts[0] : mergedFit;
    renderResults(fitTexts, downloadableFit, diagnosed, predictive, prior, truthForRecovery);
  } catch (error) {
    showRunError(error);
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

function renderResults(fitTexts, downloadableFit, diagnosed, predictive, prior, truth) {
  dashboardData = readDashboardData({ fits: fitTexts, diagnose: diagnosed.rawBytes });
  const expandedTruth = truth === null ? null : expandedRecoveryTruth(dashboardData, truth);
  recovery = expandedTruth === null ? null : recoverySummary(dashboardData, expandedTruth);
  renderChainPlot();
  element("#plot-esshat").innerHTML = renderEssRhat(dashboardData);
  element("#plot-precis").innerHTML = expandedTruth === null
    ? renderPrecis(dashboardData)
    : renderPrecis(dashboardData, expandedTruth);
  if (predictive === null) {
    element("#plot-ppc").textContent =
      "Posterior predictive display is not available for partially observed models yet.";
  } else {
    element("#plot-ppc").innerHTML = renderDensityOverlay(
      predictivePlotData(compiled.ir, boundDocument, predictive.rawBytes),
    );
  }
  if (prior === null) {
    element("#plot-overlay").textContent =
      "Prior to posterior display is not available for partially observed models yet.";
  } else {
    element("#plot-overlay").innerHTML = renderPriorPosteriorOverlay(
      overlayPlotData(dashboardData, prior.rawBytes),
    );
  }

  fitDownload = new Blob([downloadableFit], { type: "application/x-ndjson" });
  diagnosticsDownload = new Blob([diagnosed.rawBytes], { type: "application/json" });
  element("#download-fit").disabled = false;
  element("#download-diagnostics").disabled = false;
  for (const button of document.querySelectorAll(".download-svg")) {
    button.disabled =
      (predictive === null && button.dataset.plot === "ppc") ||
      (prior === null && button.dataset.plot === "overlay");
  }
  element("#results").hidden = false;
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
  const replicates = subsampleReplicates(
    ndjsonDraws(bytes).map((draw) =>
      names.flatMap((name) => arrayValue(draw.values?.[name])),
    ),
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
  element("#results").hidden = true;
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

function clearRunError() {
  runError.textContent = "";
  runError.hidden = true;
}

function showRunError(error) {
  runError.textContent = error instanceof Error ? error.message : String(error);
  runError.hidden = false;
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
