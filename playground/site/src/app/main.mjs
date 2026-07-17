import {
  MAX_DOCUMENT_INPUT_BYTES,
  MAX_DOCUMENT_SCALARS,
  parseDocument,
  parseDocumentValue,
  serializeDocument,
} from "../data/documents.mjs";
import { parseGeneratedDatasets } from "../generation/artifact.mjs";
import { evaluateDesignExpression } from "../generation/design-expr.mjs";
import {
  fixed,
  generateDatasets,
  generationInvalidationKey,
  modelPrior,
  posteriorOf,
} from "../generation/plan.mjs";
import {
  dashboardPlotLimit,
  prepareDashboardPlots,
  readDashboardData,
} from "../dashboard/index.mjs";
import { BrowserRuntime } from "../runtime/browser-runtime.mjs";
import {
  MAX_SAMPLE_CHAINS,
  MAX_SAMPLE_DRAWS,
  MAX_SAMPLE_TREEDEPTH,
  MAX_SAMPLE_WARMUP,
  MIN_SAMPLE_CHAINS,
  MIN_SAMPLE_DRAWS,
  MIN_SAMPLE_TREEDEPTH,
  MIN_SAMPLE_WARMUP,
} from "../sampling-limits.mjs";
import {
  MAX_DESIGN_FORM_SLOTS,
  MAX_PARAMETER_FORM_FIELDS,
  supportsDesignForms,
  supportsParameterForms,
} from "./form-limits.mjs";
import { recoveryTruthMap, renderRecoverySummary } from "./recovery.mjs";
import { cancellationEvents, RunControllers } from "./run-controllers.mjs";
import { assertScenarioCompatible, projectRecoveryTruth } from "./scenario.mjs";
import { decodeProject, encodeProject, FRAGMENT_WARN_LENGTH } from "./share.mjs";
import { initialState, reduce } from "./state.mjs";

const runtime = new BrowserRuntime();
const runControllers = new RunControllers();
let state = initialState();
let revision = 0;
let pendingSharedProject;
let exampleLoadRevision = 0;
let renderedSchema = null;
let designJsonMode = true;
let truthJsonMode = true;
let designExpressions = {};
let fixedValueEntries = {};
let authoringError = null;
let authoringRestore = { kind: "fresh" };
let designDocumentIsSchemaDefault = false;
let truthDocumentIsSchemaDefault = false;
const examples = new Map();
const objectUrls = new Set();
const EXAMPLES_ROOT = new URL("../../examples/", import.meta.url);

const source = element("#model-source");
const observed = element("#observed-data");
const design = element("#design-data");
const truth = element("#truth-data");
const priorSource = element("#prior-source");

source.addEventListener("input", () => {
  exampleLoadRevision += 1;
  // The old schema no longer describes schema-generated forms or placeholders.
  // Explicit JSON remains authoritative and is reconsidered after compilation.
  if (designDocumentIsSchemaDefault) {
    design.value = "";
    designDocumentIsSchemaDefault = false;
  }
  if (truthDocumentIsSchemaDefault) {
    truth.value = "";
    truthDocumentIsSchemaDefault = false;
  }
  // A shared restore keeps its carried documents authoritative (JSON-first)
  // even after a source tweak; only stale shared form state is dropped. Other
  // contexts re-derive authoring mode from the new schema.
  authoringRestore =
    authoringRestore.kind === "shared" || authoringRestore.kind === "shared-documents"
      ? { kind: "shared-documents" }
      : { kind: "fresh" };
  element("#progress").replaceChildren();
  dispatch({ type: "source-edited", source: source.value, revision: ++revision });
});
observed.addEventListener("input", observedEdited);
design.addEventListener("input", () => {
  designJsonMode = true;
  designDocumentIsSchemaDefault = false;
  authoringError = null;
  generationInputsEdited();
});
truth.addEventListener("input", () => {
  truthJsonMode = true;
  truthDocumentIsSchemaDefault = false;
  authoringError = null;
  generationInputsEdited();
});
priorSource.addEventListener("input", () => {
  authoringError = null;
  generationInputsEdited();
});
element("#design-json-toggle").addEventListener("click", toggleDesignJson);
element("#truth-json-toggle").addEventListener("click", toggleTruthJson);
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
element("#generate-button").addEventListener("click", () => launchRun(generateCollection));
element("#generated-dataset-index").addEventListener("change", () => {
  selectGeneratedPair(integerValue("#generated-dataset-index"));
});
element("#fit-button").addEventListener("click", () => {
  launchRun(selectedDatasetSource() === "generated" ? sampleGenerated : samplePosterior);
});
element("#cancel-run").addEventListener("click", cancelActiveRun);
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
  let assets;
  try {
    assets = await Promise.all([
      fetchAsset(entry.source), fetchOptionalAsset(entry.observed),
      fetchOptionalAsset(entry.design), fetchOptionalAsset(entry.truth),
    ]);
  } catch (error) {
    if (loadRevision !== exampleLoadRevision ||
        element("#examples-menu").value !== selectedId) return;
    element("#compile-error").hidden = false;
    element("#compile-error").textContent = message(error);
    return;
  }
  if (loadRevision !== exampleLoadRevision || element("#examples-menu").value !== selectedId) return;
  const [modelSource, observedText, designText, truthText] = assets;
  setProject(
    { source: modelSource, observed: observedText, design: designText, truth: truthText },
    { kind: "documents" },
  );
}

async function fetchAsset(path) {
  const response = await fetch(new URL(path, EXAMPLES_ROOT));
  if (!response.ok) throw new Error(`Could not load example asset ${path}`);
  if (response.body === null) {
    throw new Error(`Could not stream example asset ${path}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const parts = [];
  let byteLength = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      byteLength += value.byteLength;
      if (byteLength > MAX_DOCUMENT_INPUT_BYTES) {
        await reader.cancel("example asset too large");
        throw new Error(
          `Example asset ${path} exceeds ${MAX_DOCUMENT_INPUT_BYTES} UTF-8 bytes`,
        );
      }
      parts.push(decoder.decode(value, { stream: true }));
    }
    parts.push(decoder.decode());
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(`Example asset ${path} is not valid UTF-8`, { cause: error });
    }
    throw error;
  } finally {
    reader.releaseLock();
  }
  return parts.join("");
}

function fetchOptionalAsset(path) {
  return path === undefined ? Promise.resolve("") : fetchAsset(path);
}

async function shareProject() {
  const shareError = element("#share-error");
  shareError.hidden = true;
  shareError.textContent = "";
  // Authoring state describes the compiled schema; a source edit resets the
  // compile, so stale form modes and entries are omitted from the payload
  // and the recipient lands in the JSON documents instead.
  const authoringCurrent = state.compile.status === "compiled" &&
    state.compile.revision === state.sourceRevision;
  const project = {
    v: 1,
    source: source.value,
    observed: observed.value,
    design: design.value,
    truth: truth.value,
    priorSource: priorSource.value,
    sampler: samplerSettings(),
    generation: {
      seed: integerValue("#generation-seed"),
      count: integerValue("#generation-count"),
    },
    ...(authoringCurrent ? {
      authoring: {
        design: { json: designJsonMode, expressions: { ...designExpressions } },
        truth: { json: truthJsonMode, values: { ...fixedValueEntries } },
      },
    } : {}),
  };
  let payload;
  try {
    payload = await encodeProject(project);
  } catch (error) {
    element("#share-output").hidden = true;
    shareError.hidden = false;
    shareError.textContent = message(error);
    return;
  }
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
    const sharedPriorSource = String(pendingSharedProject.priorSource ?? "");
    element("#share-prior-source").textContent = sharedPriorSource;
    element("#share-prior-review").hidden = sharedPriorSource === "";
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
    prior: String(pendingSharedProject.priorSource ?? ""),
  }, validAuthoringState(pendingSharedProject.authoring)
    ? { kind: "shared", value: pendingSharedProject.authoring }
    : { kind: "shared-documents" });
  applySamplerSettings(pendingSharedProject.sampler);
  applyGenerationSettings(pendingSharedProject.generation, pendingSharedProject.sampler);
  element("#share-review").hidden = true;
}

function setProject(project, restore = { kind: "documents" }) {
  exampleLoadRevision += 1;
  element("#progress").replaceChildren();
  for (const [control, value] of [
    [source, project.source], [observed, project.observed], [design, project.design],
    [truth, project.truth], [priorSource, project.prior ?? ""],
  ]) {
    control.textContent = value;
    control.value = value;
  }
  authoringRestore = restore;
  designDocumentIsSchemaDefault = false;
  truthDocumentIsSchemaDefault = false;
  designJsonMode = true;
  truthJsonMode = true;
  authoringError = null;
  renderedSchema = null;
  dispatch({ type: "source-edited", source: project.source, revision: ++revision });
  dispatch({
    type: "documents-edited",
    documents: {
      observed: project.observed, design: project.design, truth: project.truth,
      prior: project.prior ?? "",
    },
    revision: ++revision,
  });
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

function cancelActiveRun() {
  const cancellationMessage = "Cancelled. You can start another run.";
  element("#progress").replaceChildren();
  for (const event of cancellationEvents(state, cancellationMessage)) dispatch(event);
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
  // A recipient edit before the first compile outranks a pending shared
  // authoring restore; the edited documents become the source of truth.
  if (authoringRestore.kind === "shared") authoringRestore = { kind: "shared-documents" };
  element("#progress").replaceChildren();
  dispatch({
    type: "generation-input-edited",
    documents: {
      ...state.documents, design: design.value, truth: truth.value, prior: priorSource.value,
    },
    revision: ++revision,
  });
}

function configureAuthoring(schema) {
  const restoreKind = authoringRestore.kind;
  const designFormsSupported = supportsDesignForms(schema);
  if (restoreKind === "shared") {
    const saved = authoringRestore.value;
    designJsonMode = saved.design.json;
    truthJsonMode = saved.truth.json;
    designExpressions = Object.fromEntries(
      schema.data.map((slot) => [
        slot.name, entryOr(saved.design.expressions, slot.name, defaultExpression(slot)),
      ]),
    );
    fixedValueEntries = Object.fromEntries(
      schema.parameters.map((parameter) => [
        parameter.name,
        entryOr(saved.truth.values, parameter.name, defaultParameterEntry(parameter)),
      ]),
    );
  } else if (restoreKind === "shared-documents") {
    // A legacy share or a recipient edit carries authoritative documents but
    // no current form state. Keep the PR-75 JSON-first restore behavior.
    designJsonMode = true;
    truthJsonMode = true;
    designExpressions = defaultDesignExpressions(schema);
    fixedValueEntries = defaultFixedValues(schema);
  } else if (restoreKind === "documents" || restoreKind === "fresh") {
    const designDocumentIsEmpty = design.value.trim() === "";
    designJsonMode = !designFormsSupported || !designDocumentIsEmpty;
    truthJsonMode = restoreKind === "documents" || truth.value.trim() !== "";
    designExpressions = defaultDesignExpressions(schema);
    fixedValueEntries = defaultFixedValues(schema);
    if (!designFormsSupported && designDocumentIsEmpty && canScaffoldDesign(schema)) {
      design.value = defaultDesignDocument(schema);
      designDocumentIsSchemaDefault = true;
    }
  } else {
    designExpressions = Object.fromEntries(
      schema.data.map((slot) => [
        slot.name, entryOr(designExpressions, slot.name, defaultExpression(slot)),
      ]),
    );
    fixedValueEntries = Object.fromEntries(
      schema.parameters.map((parameter) => [
        parameter.name,
        entryOr(fixedValueEntries, parameter.name, defaultParameterEntry(parameter)),
      ]),
    );
  }
  if (!designFormsSupported) designJsonMode = true;
  if (!supportsParameterForms(schema)) truthJsonMode = true;
  authoringRestore = { kind: "preserve" };
  renderedSchema = null;
  // Shared documents keep their carried bytes: they were produced by the
  // sender's evaluation, and re-evaluating seeded draws on another engine
  // could change last-bit floats. Rewrites happen only on local edits.
  // Only an initial (fresh/example) materialization is the untouched schema
  // default. The preserve path re-serializes the current form entries, which
  // may be user-authored, so it must not re-flag them as defaults — that would
  // let a later source edit drop them.
  const materializesInitialDefault = restoreKind === "documents" || restoreKind === "fresh";
  if (restoreKind !== "shared" && restoreKind !== "shared-documents" &&
      (!designJsonMode || !truthJsonMode || design.value !== state.documents.design)) {
    if (!designJsonMode) {
      writeDesignDocumentFromEntries(schema);
      if (materializesInitialDefault) designDocumentIsSchemaDefault = true;
    }
    if (!truthJsonMode) {
      writeTruthDocumentFromEntries(schema);
      if (materializesInitialDefault) truthDocumentIsSchemaDefault = true;
    }
    generationInputsEdited();
  }
}

function validAuthoringState(value) {
  return value !== null && typeof value === "object" &&
    value.design !== null && typeof value.design === "object" &&
    typeof value.design.json === "boolean" &&
    value.design.expressions !== null && typeof value.design.expressions === "object" &&
    !Array.isArray(value.design.expressions) &&
    Object.values(value.design.expressions).every((entry) => typeof entry === "string") &&
    value.truth !== null && typeof value.truth === "object" &&
    typeof value.truth.json === "boolean" &&
    value.truth.values !== null && typeof value.truth.values === "object" &&
    !Array.isArray(value.truth.values) &&
    Object.values(value.truth.values).every((entry) => typeof entry === "string");
}

function defaultDesignExpressions(schema) {
  return Object.fromEntries(schema.data.map((slot) => [slot.name, defaultExpression(slot)]));
}

const DEFAULT_DESIGN_RANGE_START = -2;
const DEFAULT_DESIGN_RANGE_STOP = 2;

function defaultExpression(slot) {
  if (slot.kind !== "vector") return "[]";
  // linspace needs n >= 2; exact lengths below that get a literal default.
  if (slot.length !== null && slot.length < 2) {
    return JSON.stringify(Array(slot.length).fill(0));
  }
  return `linspace(${DEFAULT_DESIGN_RANGE_START}, ${DEFAULT_DESIGN_RANGE_STOP}, ${slot.length ?? 25})`;
}

// A canonical document carries explicit dtype and shape, so a non-eligible
// model's placeholder honors each slot's real length and type — a plain-JSON
// placeholder cannot (JSON.parse collapses 0.0 to 0, inferring int64) and an
// empty array for a fixed-length vector would pass validation yet be
// shape-invalid.
function defaultDesignDocument(schema) {
  const variables = {};
  for (const slot of schema.data) {
    defineOwn(variables, slot.name, placeholderVariable(slot));
  }
  return JSON.stringify({ format: "bayescycle.data.json.v1", variables });
}

// A schema-shaped placeholder needs each slot's exact shape. Only scalars and
// statically-fixed-length vectors qualify; a matrix/higher-rank slot or a
// dimension-linked/unsized vector (no known length) has no scaffoldable shape,
// so no runnable placeholder is emitted and the design stays empty until the
// user authors it — an empty [] would otherwise run generation with zero rows.
// The total scaffolded scalar count is bounded by the document budget so a
// user-declared huge static length (e.g. Data.vector(1e9)) cannot allocate a
// giant placeholder during compile.
function canScaffoldDesign(schema) {
  let scalarCount = 0;
  for (const slot of schema.data) {
    if (slot.kind === "scalar") {
      scalarCount += 1;
    } else if (slot.kind === "vector" && slot.length !== null) {
      scalarCount += slot.length;
    } else {
      return false;
    }
    if (scalarCount > MAX_DOCUMENT_SCALARS) return false;
  }
  return true;
}

function placeholderVariable(slot) {
  // canScaffoldDesign gates defaultDesignDocument, so only scalars and
  // known-length vectors reach here; reject anything else rather than
  // fabricate a rank-1 or zero-length shape.
  const fill = slot.dtype === "bool" ? false : 0;
  if (slot.kind === "scalar") {
    return { dtype: slot.dtype, shape: [], values: [fill] };
  }
  if (slot.kind !== "vector" || slot.length === null) {
    throw new Error(`Cannot scaffold a ${slot.kind} design slot without a known length`);
  }
  return {
    dtype: slot.dtype,
    shape: [slot.length],
    values: Array.from({ length: slot.length }, () => fill),
  };
}

function evaluateSlotValues(slot, expression) {
  return requireSlotLength(slot, evaluateDesignExpression(expression));
}

function requireSlotLength(slot, values) {
  if (slot.length !== null && values.length !== slot.length) {
    throw new Error(`${slot.name} needs exactly ${slot.length} values, got ${values.length}`);
  }
  return values;
}

function defaultFixedValues(schema) {
  return Object.fromEntries(
    schema.parameters.map((parameter) => [parameter.name, defaultParameterEntry(parameter)]),
  );
}

function defaultParameterEntry(parameter) {
  return parameter.default === null ? "" : String(parameter.default);
}

function toggleDesignJson() {
  const schema = state.compile.modelSchema;
  if (state.compile.status !== "compiled" || schema === undefined ||
      !supportsDesignForms(schema)) return;
  if (designJsonMode) {
    if (!hydrateDesignExpressions(schema)) {
      render();
      return;
    }
    designJsonMode = false;
    commitRewrittenDocument(design, () => writeDesignDocumentFromEntries(schema));
    return;
  }
  designJsonMode = true;
  authoringError = null;
  render();
}

function toggleTruthJson() {
  const schema = state.compile.modelSchema;
  if (state.compile.status !== "compiled" || schema === undefined ||
      !supportsParameterForms(schema)) return;
  if (truthJsonMode) {
    if (!hydrateFixedValues(schema)) {
      render();
      return;
    }
    truthJsonMode = false;
    commitRewrittenDocument(truth, () => writeTruthDocumentFromEntries(schema));
    return;
  }
  truthJsonMode = true;
  authoringError = null;
  render();
}

// Entering a form mode makes the form the source of truth immediately: the
// hidden document is rewritten from the entries so a later simulate cannot
// send JSON the visible form no longer reflects.
function commitRewrittenDocument(control, write) {
  const previous = control.value;
  write();
  if (control.value === previous) {
    render();
    return;
  }
  generationInputsEdited();
}

function hydrateDesignExpressions(schema) {
  try {
    const values = plainDocumentValues(design.value);
    for (const slot of schema.data) {
      const value = values[slot.name];
      if (Array.isArray(value)) designExpressions[slot.name] = JSON.stringify(value);
    }
  } catch (error) {
    authoringError = `Design JSON: ${message(error)}`;
    return false;
  }
  authoringError = null;
  updateAuthoringControls(schema);
  return true;
}

function hydrateFixedValues(schema) {
  try {
    const values = plainDocumentValues(truth.value);
    for (const parameter of schema.parameters) {
      const value = values[parameter.name];
      if (parameter.shape.length === 0 && typeof value === "number") {
        fixedValueEntries[parameter.name] = String(value);
      } else if (parameter.shape.length !== 0 && value !== undefined) {
        fixedValueEntries[parameter.name] = JSON.stringify(value);
      }
    }
  } catch (error) {
    authoringError = `Fixed parameter JSON: ${message(error)}`;
    return false;
  }
  authoringError = null;
  updateAuthoringControls(schema);
  return true;
}

function plainDocumentValues(text) {
  const document = parseDocument(text);
  return Object.fromEntries(
    Object.entries(document.variables).map(([name, variable]) => [
      name,
      reshapeValues(variable.values, variable.shape),
    ]),
  );
}

function reshapeValues(values, shape) {
  let projectedValues = 0;
  let width = 1;
  for (const dimension of shape) {
    width *= dimension;
    projectedValues += width;
    if (!Number.isSafeInteger(projectedValues) ||
        projectedValues > MAX_DOCUMENT_SCALARS) {
      throw new Error(
        `document shape projection exceeds ${MAX_DOCUMENT_SCALARS} form values`,
      );
    }
  }
  if (shape.length === 0) return values[0];
  const stride = shape.slice(1).reduce((left, right) => left * right, 1);
  return Array.from({ length: shape[0] }, (_, index) =>
    reshapeValues(values.slice(index * stride, (index + 1) * stride), shape.slice(1)));
}

function writeDesignDocumentFromEntries(schema) {
  try {
    const value = {};
    let scalarCount = 0;
    for (const slot of schema.data) {
      const values = evaluateSlotValues(
        slot,
        entryOr(designExpressions, slot.name, defaultExpression(slot)),
      );
      scalarCount += values.length;
      requireScalarBudget(scalarCount);
      defineOwn(value, slot.name, values);
    }
    design.value = serializeDocument(parseDocument(JSON.stringify(value)));
    authoringError = null;
    return true;
  } catch (error) {
    design.value = "";
    authoringError = message(error);
    return false;
  }
}

function writeTruthDocumentFromEntries(schema) {
  try {
    const value = {};
    let scalarCount = 0;
    for (const parameter of schema.parameters) {
      const entry = entryOr(fixedValueEntries, parameter.name, "");
      let parsed;
      if (parameter.shape.length === 0) {
        parsed = Number(entry);
        if (entry.trim() === "" || !Number.isFinite(parsed)) {
          throw new Error(`${parameter.name} needs a finite number`);
        }
      } else {
        parsed = parseDocumentValue(entry);
        if (!Array.isArray(parsed)) {
          throw new Error(`${parameter.name} needs a JSON array value`);
        }
      }
      scalarCount += countScalars(parsed);
      requireScalarBudget(scalarCount);
      defineOwn(value, parameter.name, parsed);
    }
    truth.value = serializeDocument(parseDocument(JSON.stringify(value)));
    authoringError = null;
    return true;
  } catch (error) {
    truth.value = "";
    authoringError = message(error);
    return false;
  }
}

function defineOwn(target, name, value) {
  Object.defineProperty(target, name, {
    value,
    enumerable: true,
    writable: true,
    configurable: true,
  });
}

function requireScalarBudget(count) {
  if (count > MAX_DOCUMENT_SCALARS) {
    throw new Error(
      `data document exceeds maximum scalar count of ${MAX_DOCUMENT_SCALARS}`,
    );
  }
}

function countScalars(value) {
  if (!Array.isArray(value)) return 1;
  let count = 0;
  const arrays = [value];
  const indexes = [0];
  while (arrays.length > 0) {
    const depth = arrays.length - 1;
    const array = arrays[depth];
    if (indexes[depth] >= array.length) {
      arrays.pop();
      indexes.pop();
      continue;
    }
    const entry = array[indexes[depth]++];
    if (Array.isArray(entry)) {
      arrays.push(entry);
      indexes.push(0);
    } else {
      count += 1;
    }
  }
  return count;
}

function renderAuthoring() {
  const schema = state.compile.status === "compiled" ? state.compile.modelSchema : undefined;
  if (schema === undefined) {
    renderedSchema = null;
    element("#design-slots").innerHTML = '<p class="hint">Compile a model to author its design inputs.</p>';
    element("#parameter-fields").innerHTML = '<p class="hint">Compile a model to author fixed parameter values.</p>';
    element("#design-json-field").hidden = false;
    element("#truth-json-field").hidden = false;
    element("#design-json-toggle").disabled = true;
    element("#truth-json-toggle").disabled = true;
    return;
  }
  if (renderedSchema !== schema) {
    buildDesignSlots(schema);
    buildParameterFields(schema);
    renderedSchema = schema;
  }
  element("#design-json-field").hidden = !designJsonMode;
  element("#design-slots").hidden = designJsonMode;
  const designFormsSupported = supportsDesignForms(schema);
  element("#design-json-toggle").disabled = !designFormsSupported;
  element("#design-json-toggle").textContent = !designFormsSupported
    ? schema.data.length > MAX_DESIGN_FORM_SLOTS
      ? `JSON required for more than ${MAX_DESIGN_FORM_SLOTS} design slots`
      : "JSON required for integer or non-vector slots"
    : designJsonMode ? "use design forms" : "edit as JSON";
  element("#truth-json-field").hidden = !truthJsonMode;
  element("#parameter-fields").hidden = truthJsonMode;
  const parameterFormsSupported = supportsParameterForms(schema);
  element("#truth-json-toggle").disabled = !parameterFormsSupported;
  element("#truth-json-toggle").textContent = parameterFormsSupported
    ? truthJsonMode ? "use parameter form" : "edit as JSON"
    : `JSON required for more than ${MAX_PARAMETER_FORM_FIELDS} parameters`;
  if (!designJsonMode) renderDesignPreviews(schema);
  renderPlanSummary();
}

function buildDesignSlots(schema) {
  const container = element("#design-slots");
  container.replaceChildren();
  if (schema.data.length > MAX_DESIGN_FORM_SLOTS) {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = `JSON required for more than ${MAX_DESIGN_FORM_SLOTS} design slots.`;
    container.append(hint);
    return;
  }
  if (schema.data.length === 0) {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "This model has no design inputs.";
    container.append(hint);
    return;
  }
  for (const [index, slot] of schema.data.entries()) {
    const card = document.createElement("div");
    card.className = "slot-card";
    const head = document.createElement("div");
    head.className = "slot-head";
    const name = document.createElement("span");
    name.className = "slot-name";
    name.textContent = slot.name;
    const type = document.createElement("span");
    type.className = "slot-type";
    type.textContent = `${slot.kind} · ${slot.dtype}`;
    head.append(name, type);
    const row = document.createElement("div");
    row.className = "slot-expr";
    const equals = document.createElement("span");
    equals.className = "eq";
    equals.textContent = "=";
    const input = document.createElement("input");
    input.id = `design-expr-${safeId(slot.name)}`;
    input.type = "text";
    input.spellcheck = false;
    input.autocomplete = "off";
    input.value = entryOr(designExpressions, slot.name, defaultExpression(slot));
    input.setAttribute("aria-label", `Design expression for ${slot.name}`);
    input.dataset.slotIndex = String(index);
    input.addEventListener("input", () => {
      designExpressions[slot.name] = input.value;
      syncDesignForms();
    });
    row.append(equals, input);
    const vocabulary = document.createElement("p");
    vocabulary.className = "slot-vocab";
    vocabulary.textContent = "Vocabulary: linspace(start, stop, n) · repeat([v, …], times) · normal(loc, scale, n, seed=<int>) · uniform(low, high, n, seed=<int>) · literal [v, v, …]";
    const preview = document.createElement("pre");
    preview.id = `design-preview-${safeId(slot.name)}`;
    preview.className = "slot-preview";
    card.append(head, row, vocabulary, preview);
    container.append(card);
  }
}

function buildParameterFields(schema) {
  const container = element("#parameter-fields");
  container.replaceChildren();
  if (!supportsParameterForms(schema)) {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = `JSON required for more than ${MAX_PARAMETER_FORM_FIELDS} parameters.`;
    container.append(hint);
    return;
  }
  if (schema.parameters.length === 0) {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "This model has no fixed parameters.";
    container.append(hint);
    return;
  }
  const grid = document.createElement("div");
  grid.className = "param-grid";
  for (const parameter of schema.parameters) {
    const label = document.createElement("span");
    label.className = "param-label";
    const name = document.createElement("span");
    name.className = "param-name";
    name.textContent = parameter.name;
    const prior = document.createElement("span");
    prior.className = "param-prior";
    prior.textContent = parameter.prior;
    label.append(name, prior);
    if (parameter.constraint !== null) {
      const constraint = document.createElement("span");
      constraint.className = "constraint";
      constraint.textContent = parameter.constraint;
      label.append(constraint);
    }
    const input = document.createElement("input");
    input.id = `fixed-value-${safeId(parameter.name)}`;
    input.value = entryOr(
      fixedValueEntries, parameter.name, defaultParameterEntry(parameter),
    );
    input.type = parameter.shape.length === 0 ? "number" : "text";
    input.step = parameter.shape.length === 0 ? "any" : "";
    input.setAttribute("aria-label", `${parameter.name} fixed value`);
    if (parameter.shape.length !== 0) {
      input.placeholder = `JSON value with shape [${parameter.shape.join(", ")}]`;
    }
    input.addEventListener("input", () => {
      fixedValueEntries[parameter.name] = input.value;
      syncTruthForm();
    });
    grid.append(label, input);
  }
  container.append(grid);
}

function updateAuthoringControls(schema) {
  for (const slot of schema.data) {
    const input = document.getElementById(`design-expr-${safeId(slot.name)}`);
    if (input !== null) {
      input.value = entryOr(designExpressions, slot.name, defaultExpression(slot));
    }
  }
  for (const parameter of schema.parameters) {
    const input = document.getElementById(`fixed-value-${safeId(parameter.name)}`);
    if (input !== null) input.value = entryOr(fixedValueEntries, parameter.name, "");
  }
}

function syncDesignForms() {
  const schema = state.compile.modelSchema;
  if (state.compile.status !== "compiled" || schema === undefined) return;
  // A form edit makes the design user-authored, so a later source edit must
  // preserve it rather than discard it as the untouched schema default.
  designDocumentIsSchemaDefault = false;
  writeDesignDocumentFromEntries(schema);
  renderDesignPreviews(schema);
  generationInputsEdited();
}

function syncTruthForm() {
  const schema = state.compile.modelSchema;
  if (state.compile.status !== "compiled" || schema === undefined) return;
  // A form edit makes the fixed values user-authored, so a later source edit
  // preserves them rather than discarding them as the schema default.
  truthDocumentIsSchemaDefault = false;
  writeTruthDocumentFromEntries(schema);
  generationInputsEdited();
}

function renderDesignPreviews(schema) {
  let previewScalarCount = 0;
  let previewBudgetExceeded = false;
  for (const slot of schema.data) {
    const input = document.getElementById(`design-expr-${safeId(slot.name)}`);
    const preview = document.getElementById(`design-preview-${safeId(slot.name)}`);
    if (input === null || preview === null) continue;
    if (previewBudgetExceeded) {
      input.classList.remove("invalid");
      preview.classList.remove("invalid");
      preview.textContent = "preview omitted (too large)";
      continue;
    }
    try {
      const values = evaluateDesignExpression(input.value);
      previewScalarCount += values.length;
      if (previewScalarCount > MAX_DOCUMENT_SCALARS) {
        previewBudgetExceeded = true;
        input.classList.remove("invalid");
        preview.classList.remove("invalid");
        preview.textContent = "preview omitted (too large)";
        continue;
      }
      requireSlotLength(slot, values);
      const shown = values.slice(0, 8).map((value) => Number(value).toFixed(2)).join(", ");
      input.classList.remove("invalid");
      preview.classList.remove("invalid");
      preview.textContent = `${slot.name} ← shape [${values.length}] · [${shown}${values.length > 8 ? ", …" : ""}]`;
    } catch (error) {
      input.classList.add("invalid");
      preview.classList.add("invalid");
      preview.textContent = `✗ ${message(error)}`;
    }
  }
}

function renderPlanSummary() {
  const count = element("#generation-count").value;
  const seed = element("#generation-seed").value;
  const parameterSource = {
    fixed: "fixed values",
    prior: "model prior",
    "other-prior": "another prior composed with the model",
    posterior: "posterior from fit",
  }[selectedParamSource()];
  const designSource = designJsonMode
    ? "design JSON"
    : Object.entries(designExpressions).map(([name, expression]) => `${name}: ${expression}`).join(", ");
  element("#plan-summary").textContent =
    `draw(count=${count}, seed=${seed})\n  parameters: ${parameterSource}\n  design: ${designSource || "{}"}`;
}

function entryOr(entries, name, fallback) {
  return Object.hasOwn(entries, name) ? entries[name] : fallback;
}

function safeId(name) {
  return [...name].map((character) => /[A-Za-z0-9_-]/.test(character)
    ? character
    : `-${character.codePointAt(0).toString(16)}-`).join("");
}

async function compileModel() {
  const requestId = crypto.randomUUID();
  const sourceRevision = state.sourceRevision;
  const priorIrHash = state.compile.status === "compiled"
    ? state.compile.irHash
    : state.compile.priorIrHash ?? null;
  let completionRevision = sourceRevision;
  const controller = runControllers.begin("compile", requestId);
  dispatch({ type: "compile-started", requestId, revision: sourceRevision });
  if (state.compile.status !== "compiling" || state.compile.requestId !== requestId) return;
  try {
    const result = await runtime.compile(state.source, { signal: controller.signal });
    if (result.ok) {
      if (state.compile.status !== "compiling" || state.compile.requestId !== requestId ||
          state.sourceRevision !== sourceRevision) return;
      // A fresh interpreter can legitimately produce different bytes for the
      // same source. Reuse the reducer's source invalidation transition so no
      // fit or generation remains associated with the prior model bytes.
      if (priorIrHash !== null && priorIrHash !== result.irHash) {
        completionRevision = ++revision;
        dispatch({
          type: "source-edited", source: state.source, revision: completionRevision,
        });
        dispatch({
          type: "compile-started", requestId, revision: completionRevision,
        });
      }
      configureAuthoring(result.modelSchema);
      dispatch({
        type: "compile-succeeded", requestId, revision: completionRevision,
        irBytes: result.irBytes, irHash: result.irHash, modelSchema: result.modelSchema,
      });
    } else {
      dispatch({ type: "compile-failed", requestId, revision: completionRevision, error: result.traceback });
    }
  } catch (error) {
    dispatch({ type: "compile-failed", requestId, revision: completionRevision, error: message(error) });
  } finally {
    runControllers.finish("compile", requestId);
  }
}

async function samplePosterior() {
  const dataBytes = documentBytes(observed.value);
  await sampleData(dataBytes, "observed");
}

async function generateCollection() {
  if (state.compile.status !== "compiled" || state.run.status === "running" ||
      state.generation.attempt.status === "running") return;
  const originalCompile = state.compile;
  const originalModelBytes = compiledBytes();
  const originalModelHash = `sha256:${originalCompile.irHash}`;
  const mainSource = state.source;
  const currentPriorSource = priorSource.value;
  const designBytes = documentBytes(design.value);
  const sourceKind = selectedParamSource();
  const settings = generationSettings();
  let generationModelBytes = originalModelBytes;
  let parameterSource;
  let sourceFitLineageKey;
  const guardSourceKind = ["prior", "other-prior"].includes(sourceKind)
    ? "model-prior"
    : sourceKind;
  const guard = {
    compileRevision: state.compile.revision,
    inputRevision: state.generation.inputRevision,
    settingsRevision: state.generation.settingsRevision,
    sourceKind: guardSourceKind,
    fitLineageKey: null,
    priorSource: currentPriorSource,
  };

  let requestId;
  let dependencyKey;
  let controller;
  if (sourceKind === "other-prior") {
    requestId = crypto.randomUUID();
    // The disposable scenario compile is owned by the generation attempt.
    // Its request identity is sufficient until the immutable plan exists.
    dependencyKey = `scenario:${requestId}`;
    dispatch({ type: "generation-started", requestId, dependencyKey, guard });
    if (state.generation.attempt.requestId !== requestId) return;
    controller = runControllers.begin("generation", requestId);
    try {
      const composed = await runtime.compileScenario(mainSource, currentPriorSource, {
        signal: controller.signal,
      });
      if (!composed.ok) throw new Error(composed.traceback);
      if (state.generation.attempt.requestId !== requestId) return;
      assertScenarioCompatible(originalCompile, composed);
      generationModelBytes = composed.irBytes;
      parameterSource = modelPrior(generationModelBytes, {
        claimedSourceModelHash: `sha256:${composed.irHash}`,
        claimedOutcomeModelHash: originalModelHash,
      });
    } catch (error) {
      dispatch({
        type: "generation-failed", requestId, dependencyKey, error: message(error),
      });
      runControllers.finish("generation", requestId);
      return;
    }
  } else if (sourceKind === "fixed") {
    parameterSource = fixed(documentBytes(truth.value));
  } else if (sourceKind === "prior") {
    parameterSource = modelPrior(originalModelBytes);
  } else {
    const fit = state.conditioning.fit;
    if (fit === null || fit.fitArtifact === undefined) return;
    parameterSource = posteriorOf(fit.fitArtifact);
    sourceFitLineageKey = fit.lineageKey;
    guard.fitLineageKey = sourceFitLineageKey;
  }

  const plan = generateDatasets(generationModelBytes, {
    design: designBytes,
    parameterSource,
    count: settings.count,
    seed: settings.seed,
  });
  if (requestId === undefined) {
    dependencyKey = await generationInvalidationKey(plan);
    requestId = crypto.randomUUID();
    dispatch({ type: "generation-started", requestId, dependencyKey, guard });
    if (state.generation.attempt.requestId !== requestId) return;
    controller = runControllers.begin("generation", requestId);
  }

  try {
    const result = await runtime.run({
      type: "run", id: requestId, operation: "generate", plan,
    }, undefined, { signal: controller.signal });
    const artifact = result.artifacts.find(
      (entry) => entry.name === "generated_datasets.ndjson",
    );
    if (artifact === undefined) throw new Error("Runtime returned no generated collection");
    const parsed = parseGeneratedDatasets(artifact.bytes);
    const selected = parsed.select(0);
    dispatch({
      type: "generation-succeeded", requestId, dependencyKey,
      collection: {
        sourceKind: guardSourceKind,
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
  } finally {
    runControllers.finish("generation", requestId);
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
  if (selected === null || state.compile.status !== "compiled") return;
  const recovery = projectRecoveryTruth(selected.parametersBytes, state.compile.modelSchema);
  await sampleData(
    selected.datasetBytes,
    "generated",
    recovery.bytes ?? undefined,
    recovery.notice,
  );
}

async function sampleData(dataBytes, datasetSource, recoveryTruth, recoveryNotice = null) {
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
  if (state.run.status !== "running" || state.run.requestId !== requestId) {
    dispatch({
      type: "conditioning-failed",
      requestId,
      dependencyKey,
      error: "Conditioning request became stale before launch",
    });
    return;
  }
  const controller = runControllers.begin("conditioning", requestId);
  runControllers.reconcile(state);
  if (controller.signal.aborted) return;
  try {
    const sampled = await runtime.run({
      type: "run",
      id: requestId,
      operation: "condition",
      modelIr: state.compile.irBytes,
      data: dataBytes,
      settings,
      ...(recoveryTruth === undefined ? {} : { pairedParameters: recoveryTruth }),
    }, (event) => renderActiveProgress(requestId, projectRevision, event), {
      signal: controller.signal,
    });
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
      notice: [sampled.notice, recoveryNotice].filter((entry) => entry !== null).join("\n") || null,
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
  } finally {
    runControllers.finish("conditioning", requestId);
  }
}

function compiledBytes() {
  if (state.compile.status !== "compiled") throw new Error("Compile the model first");
  return state.compile.irBytes;
}

function documentBytes(text) {
  return new TextEncoder().encode(serializeDocument(parseDocument(text)));
}

function validDocument(text) {
  if (text.trim() === "") return false;
  try {
    documentBytes(text);
    return true;
  } catch {
    return false;
  }
}

function observedDocumentBindsSchema(text, schema) {
  if (text.trim() === "") return false;
  try {
    const document = parseDocument(text);
    const slots = [...schema.data, ...schema.observed];
    const expected = new Set(slots.map((slot) => slot.name));
    // An unexpected variable is rejected at binding, so the cue must not claim
    // readiness for a document whose names do not match the schema exactly.
    for (const name of Object.keys(document.variables)) {
      if (!expected.has(name)) return false;
    }
    for (const slot of slots) {
      if (!Object.hasOwn(document.variables, slot.name)) return false;
      const variable = document.variables[slot.name];
      if (!observedShapeMatchesSlot(variable.shape, slot)) return false;
      if (!observedDtypeMatchesSlot(variable.dtype, slot)) return false;
    }
    return true;
  } catch {
    return false;
  }
}

// Validate the shape constraints the schema actually knows — scalars and
// statically-fixed-length vectors. Dimension-linked vectors, observed slots,
// and higher-rank slots carry no concrete shape here; the engine validates
// those during binding.
function observedShapeMatchesSlot(shape, slot) {
  if (slot.kind === "scalar") return shape.length === 0;
  if (slot.kind === "vector" && slot.length !== null) {
    return shape.length === 1 && shape[0] === slot.length;
  }
  return true;
}

const INTEGER_DTYPES = ["int32", "int64"];
const FLOAT_DTYPES = ["float32", "float64"];

// A data slot's schema dtype constrains what binds: an integer slot rejects
// float observed values, a float slot accepts integer or float (the engine
// coerces integers up), and bool requires bool. Observed slots carry no dtype
// in the schema, so they are left to the engine.
function observedDtypeMatchesSlot(observedDtype, slot) {
  if (slot.dtype === undefined) return true;
  if (slot.dtype === "bool") return observedDtype === "bool";
  if (INTEGER_DTYPES.includes(slot.dtype)) return INTEGER_DTYPES.includes(observedDtype);
  return INTEGER_DTYPES.includes(observedDtype) || FLOAT_DTYPES.includes(observedDtype);
}

function rawGenerationDocumentError(paramSource) {
  if (state.compile.status !== "compiled") return null;
  for (const [label, enabled, text] of [
    ["Design JSON", designJsonMode, design.value],
    ["Fixed parameter JSON", paramSource === "fixed" && truthJsonMode, truth.value],
  ]) {
    if (!enabled || text.trim() === "") continue;
    try {
      documentBytes(text);
    } catch (error) {
      return `${label}: ${message(error)}`;
    }
  }
  return null;
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

// Upper bounds keep one submission tab-safe: each chain launches its own
// wasm worker, and the engine itself enforces max_treedepth 1..20.
function validSampleSettings(settings = samplerSettings()) {
  return Number.isSafeInteger(settings.chains) &&
    settings.chains >= MIN_SAMPLE_CHAINS && settings.chains <= MAX_SAMPLE_CHAINS &&
    Number.isSafeInteger(settings.num_warmup) &&
    settings.num_warmup >= MIN_SAMPLE_WARMUP &&
    settings.num_warmup <= MAX_SAMPLE_WARMUP &&
    Number.isSafeInteger(settings.num_draws) &&
    settings.num_draws >= MIN_SAMPLE_DRAWS && settings.num_draws <= MAX_SAMPLE_DRAWS &&
    validSeedSettings(settings) &&
    Number.isSafeInteger(settings.max_treedepth) &&
    settings.max_treedepth >= MIN_SAMPLE_TREEDEPTH &&
    settings.max_treedepth <= MAX_SAMPLE_TREEDEPTH &&
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
  runControllers.reconcile(state);
  render();
}

function render() {
  const hash = element("#ir-hash");
  hash.hidden = state.compile.status !== "compiled";
  hash.textContent = state.compile.status === "compiled" ? `sha256:${state.compile.irHash}` : "";
  element("#compile-status").textContent = state.compile.status === "compiling"
    ? "Compiling in an isolated worker…"
    : state.compile.status === "compiled"
      ? `Compiled. ${countLabel(state.compile.modelSchema.parameters.length, "parameter")}, ${countLabel(state.compile.modelSchema.data.length, "design slot")}, ${countLabel(state.compile.modelSchema.observed.length, "observed slot")}.`
      : "";
  const compileError = element("#compile-error");
  compileError.hidden = state.compile.status !== "failed";
  compileError.textContent = state.compile.status === "failed" ? state.compile.error : "";
  element("#compile-button").disabled = state.compile.status === "compiling" ||
    state.run.status === "running" || state.generation.attempt.status === "running" ||
    state.source.trim() === "";
  element("#share-button").disabled = state.source.trim() === "";
  const visibleArtifacts = [
    ...generationDisplayArtifacts(state.generation.collection?.artifacts ?? []),
    ...state.artifacts,
  ];
  const hasArtifact = (name) => visibleArtifacts.some((artifact) => artifact.name === name);
  const posteriorAvailable = hasArtifact("posterior.ndjson") && hasArtifact("data.json");
  const posteriorSource = element("#param-source-posterior");
  posteriorSource.disabled = !posteriorAvailable;
  element("#posterior-source-hint").hidden = posteriorAvailable;
  const otherPriorSource = element("#param-source-other-prior");
  otherPriorSource.disabled = state.compile.status !== "compiled";
  if ((posteriorSource.checked && posteriorSource.disabled) ||
      (otherPriorSource.checked && otherPriorSource.disabled)) {
    element("#param-source-fixed").checked = true;
  }

  renderAuthoring();
  renderGenerationSelection();
  const generatedAvailable = state.generation.selected !== null;
  const generatedSource = element("#dataset-source-generated");
  generatedSource.disabled = !generatedAvailable;
  if (generatedSource.checked && generatedSource.disabled) element("#dataset-source-observed").checked = true;

  const paramSource = selectedParamSource();
  const datasetSource = selectedDatasetSource();
  element("#fixed-values-field").hidden = paramSource !== "fixed";
  element("#other-prior-field").hidden = paramSource !== "other-prior";
  const count = integerValue("#generation-count");
  const countText = Number.isSafeInteger(count) && count >= 1 ? String(count) : "N";
  const datasetWord = count === 1 ? "dataset" : "datasets";
  element("#generate-button").textContent = {
    fixed: `Simulate ${countText} ${datasetWord} at fixed values`,
    prior: `Simulate ${countText} ${datasetWord} from the model prior`,
    "other-prior": `Simulate ${countText} ${datasetWord} from the composed prior`,
    posterior: `Simulate ${countText} ${datasetWord} from the posterior`,
  }[paramSource];
  const selectedIndex = state.generation.selected?.index ?? 0;
  element("#fit-button").textContent = datasetSource === "generated"
    ? `Fit simulated pair ${selectedIndex}`
    : "Fit observed data";
  element("#observed-data-field").hidden = datasetSource !== "observed";

  const unavailable = state.compile.status !== "compiled" || state.run.status === "running" ||
    state.generation.attempt.status === "running";
  element("#generate-button").disabled = unavailable || !validGenerationSeed() ||
    !validGenerationCount() ||
    !validDocument(design.value) ||
    (paramSource === "fixed" && !validDocument(truth.value)) ||
    (paramSource === "other-prior" && priorSource.value.trim() === "") ||
    (paramSource === "posterior" && !posteriorAvailable);
  element("#fit-button").disabled = unavailable || !validSampleSettings() ||
    (datasetSource === "observed" && observed.value.trim() === "") ||
    (datasetSource === "generated" && !generatedAvailable);
  element("#observed-ready-hint").hidden = state.compile.status !== "compiled" ||
    datasetSource !== "observed" || !validSampleSettings() ||
    !observedDocumentBindsSchema(observed.value, state.compile.modelSchema) ||
    state.run.status === "running" || state.generation.attempt.status === "running";
  const visibleAuthoringError = authoringError ?? rawGenerationDocumentError(paramSource);
  const authoringErrorElement = element("#authoring-error");
  authoringErrorElement.hidden = visibleAuthoringError === null;
  authoringErrorElement.textContent = visibleAuthoringError ?? "";
  const activeRun = state.compile.status === "compiling" ||
    state.generation.attempt.status === "running" || state.run.status === "running";
  element("#cancel-run").hidden = !activeRun;
  element("#run-status").textContent = state.generation.attempt.status === "running"
    ? state.generation.attempt.sourceKind === "model-prior" &&
        selectedParamSource() === "other-prior"
      ? "Compiling the composed prior and generating paired datasets…"
      : "Generating paired datasets is running…"
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
      option.textContent = `draw ${index}`;
      return option;
    }));
  }
  const selected = state.generation.selected;
  selector.disabled = selected === null;
  if (selected !== null) selector.value = String(selected.index);
  summary.textContent = selected === null
    ? "Select a simulated parameter/dataset pair."
    : `Pair ${selected.index} of ${collection.parsed.count} selected. Paired parameters retained for recovery.`;
  container.hidden = false;
}

function generationDisplayArtifacts(artifacts) {
  return artifacts.map((artifact) => artifact.name === "model.ir.json"
    ? { ...artifact, name: "generation:model.ir.json" }
    : artifact);
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
    "generation:model.ir.json": "#artifact-generation-model",
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
  // The generation run's manifest references model.ir.json; when both model
  // artifacts are visible the fit model downloads under its own name so a
  // flat downloads directory cannot shadow the manifested bytes.
  const fitModelLink = element("#artifact-model a");
  const fitDownloadName = element("#artifact-generation-model").hidden
    ? "model.ir.json"
    : "fit-model.ir.json";
  fitModelLink.download = fitDownloadName;
  fitModelLink.textContent = fitDownloadName;
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
  const truncatedNotice = element("#plots-truncated-notice");
  truncatedNotice.hidden = true;
  if (posterior === undefined || diagnostics === undefined) {
    plotGrid.hidden = true;
    plots.hidden = !artifacts.some((artifact) => artifact.name === "recovery_check.json");
    clearDashboardPlots();
    return;
  }
  try {
    const data = readDashboardData({ fits: [posterior.bytes], diagnose: diagnostics.bytes });
    const plotLimit = dashboardPlotLimit(data);
    if (plotLimit.kind === "omitted") {
      clearDashboardPlots();
      truncatedNotice.textContent = plotLimit.notice;
      truncatedNotice.hidden = false;
      plotGrid.hidden = true;
      plots.hidden = false;
      return;
    }
    const recovery = artifacts.find((artifact) => artifact.name === "recovery_check.json");
    const truth = recovery === undefined
      ? undefined
      : recoveryTruthMap(
          JSON.parse(new TextDecoder().decode(recovery.bytes)),
          data.parameters.map((parameter) => parameter.label),
        );
    const rendered = prepareDashboardPlots(data, truth);
    if (rendered.kind !== "rendered") throw new Error("Dashboard plot limit changed");
    element("#plot-trank").innerHTML = rendered.trank;
    element("#plot-ess-rhat").innerHTML = rendered.essRhat;
    element("#plot-precis").innerHTML = rendered.precis;
    plotGrid.hidden = false;
    plots.hidden = false;
  } catch (error) {
    clearDashboardPlots();
    plotGrid.hidden = true;
    plots.hidden = !artifacts.some((artifact) => artifact.name === "recovery_check.json");
    element("#run-error").hidden = false;
    element("#run-error").textContent = `Artifacts downloaded, but plots could not render: ${message(error)}`;
  }
}

function clearDashboardPlots() {
  for (const selector of ["#plot-trank", "#plot-ess-rhat", "#plot-precis"]) {
    element(selector).replaceChildren();
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

// A cleared field is invalid input, not an implicit zero.
function integerValue(selector) { return numberValue(selector); }
function numberValue(selector) {
  const text = element(selector).value.trim();
  return text === "" ? NaN : Number(text);
}
function countLabel(count, singular) { return `${count} ${singular}${count === 1 ? "" : "s"}`; }
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
