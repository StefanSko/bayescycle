import { normalizeDocument } from "../data/documents.mjs";
import {
  PortablePosteriorError,
  validatePortablePosterior,
} from "./posterior-source.mjs";

export const MAX_GENERATED_ARTIFACT_BYTES = 64 * 1024 * 1024;
export const MAX_GENERATED_LINE_BYTES = 8 * 1024 * 1024;

const MAX_DEPTH = 64;
const MAX_COUNT = 1000;
const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER;
const HASH = /^sha256:[0-9a-f]{64}$/u;
const UTF8 = new TextDecoder("utf-8", { fatal: true });
const MARKER_KEYS = [
  "generated_datasets_format",
  "artifact_kind",
  "artifact_scope",
];
const MARKER_VALUES = [
  "v0-provisional",
  "generated_dataset_pairs",
  "parameter_and_complete_dataset_joint_draws",
];
const PHASES = [
  "parse_json",
  "decode_ir",
  "bind_design",
  "draw_parameters",
  "simulate_outcomes",
  "emit_artifact",
];
const HEADER_KEYS = [
  ...MARKER_KEYS,
  "workflow_phases",
  "generation_model_hash",
  "design_hash",
  "parameter_source",
  "count",
  "seed",
  "draw_index_base",
  "parameter_schema",
  "dataset_schema",
];
const DRAW_KEYS = [
  ...MARKER_KEYS,
  "draw_index",
  "draw_count",
  "parameters",
  "dataset",
  "source_lineage",
];
const TRAILER_KEYS = [
  ...MARKER_KEYS,
  "workflow_phases",
  "generation_model_hash",
  "design_hash",
  "parameter_source",
  "count",
  "seed",
  "draw_count",
  "complete",
];

export class GeneratedDatasetsArtifactError extends Error {
  constructor(message) {
    super(message);
    this.name = "GeneratedDatasetsArtifactError";
  }
}

/** @param {Uint8Array} input */
export function parseGeneratedDatasets(input) {
  if (!(input instanceof Uint8Array)) {
    throw new GeneratedDatasetsArtifactError("generated-dataset artifact must be bytes");
  }
  const sourceBytes = Uint8Array.from(input);
  if (sourceBytes.byteLength > MAX_GENERATED_ARTIFACT_BYTES) {
    throw new GeneratedDatasetsArtifactError("generated-dataset artifact exceeds byte limit");
  }
  if (sourceBytes.at(-1) !== 0x0a) {
    throw new GeneratedDatasetsArtifactError("generated-dataset artifact needs a trailer and final LF");
  }
  const lines = splitLines(sourceBytes);
  if (lines.length < 3) {
    throw new GeneratedDatasetsArtifactError("generated-dataset artifact is missing its trailer");
  }
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line.byteLength === 0) {
      throw new GeneratedDatasetsArtifactError(`generated-dataset line ${index + 1} is empty`);
    }
    if (line.byteLength + 1 > MAX_GENERATED_LINE_BYTES) {
      throw new GeneratedDatasetsArtifactError(`generated-dataset line ${index + 1} exceeds line limit`);
    }
    validateDepth(line, index + 1);
  }
  const documents = lines.map((line, index) => parseLine(line, index + 1));
  const header = object(documents[0], "header");
  exactKeys(header, HEADER_KEYS, "header");
  marker(header, "header");
  phases(header.workflow_phases, "header");
  const generationModelHash = hash(header.generation_model_hash, "generation_model_hash");
  const designHash = hash(header.design_hash, "design_hash");
  const source = parameterSource(header.parameter_source);
  const count = integer(header.count, "count", 1, MAX_COUNT);
  const seed = integer(header.seed, "seed", 0, MAX_SAFE_INTEGER);
  if (header.draw_index_base !== "zero_based_generation_order") {
    throw new GeneratedDatasetsArtifactError("header draw_index_base is invalid");
  }
  const parameterSchema = schema(header.parameter_schema, "parameter_schema");
  const datasetSchema = schema(header.dataset_schema, "dataset_schema");

  const trailerEnvelope = object(documents.at(-1), "trailer envelope");
  exactKeys(trailerEnvelope, ["trailer"], "trailer envelope");
  const trailer = object(trailerEnvelope.trailer, "trailer");
  exactKeys(trailer, TRAILER_KEYS, "trailer");
  marker(trailer, "trailer");
  phases(trailer.workflow_phases, "trailer");
  if (
    trailer.generation_model_hash !== generationModelHash ||
    trailer.design_hash !== designHash ||
    JSON.stringify(trailer.parameter_source) !== JSON.stringify(header.parameter_source) ||
    trailer.count !== count || trailer.seed !== seed || trailer.draw_count !== count ||
    trailer.complete !== true
  ) {
    throw new GeneratedDatasetsArtifactError("header and trailer count, identity, or source disagree");
  }

  const drawDocuments = documents.slice(1, -1);
  if (drawDocuments.length !== count) {
    throw new GeneratedDatasetsArtifactError(
      `generated-dataset count declares ${count}, stream contains ${drawDocuments.length} draws`,
    );
  }
  const selections = [];
  const draws = drawDocuments.map((value, drawIndex) => {
    const draw = object(value, `draw ${drawIndex}`);
    exactKeys(draw, DRAW_KEYS, `draw ${drawIndex}`);
    marker(draw, `draw ${drawIndex}`);
    if (draw.draw_index !== drawIndex) {
      throw new GeneratedDatasetsArtifactError(
        `draw_index must be contiguous; expected ${drawIndex}, got ${String(draw.draw_index)}`,
      );
    }
    if (draw.draw_count !== count) {
      throw new GeneratedDatasetsArtifactError(`draw ${drawIndex} draw_count disagrees with header`);
    }
    const parameters = canonicalDocument(draw.parameters, `draw ${drawIndex}.parameters`);
    const dataset = canonicalDocument(draw.dataset, `draw ${drawIndex}.dataset`);
    matchesSchema(parameters, parameterSchema, `draw ${drawIndex}.parameters`);
    matchesSchema(dataset, datasetSchema, `draw ${drawIndex}.dataset`);
    const sourceLineage = lineage(draw.source_lineage, source.kind, drawIndex);
    const line = lines[drawIndex + 1];
    const parameterSpan = topLevelValueSpan(line, "parameters", 0);
    const datasetSpan = topLevelValueSpan(line, "dataset", parameterSpan.end);
    selections.push(Object.freeze({
      parametersBytes: withLf(parameterSpan.bytes),
      datasetBytes: withLf(datasetSpan.bytes),
    }));
    return deepFreeze({
      drawIndex,
      parameters,
      dataset,
      sourceKind: sourceLineage.kind,
      sourceLineage,
    });
  });

  const artifact = {
    generationModelHash,
    designHash,
    sourceKind: source.kind,
    count,
    seed,
    draws: Object.freeze(draws),
    get bytes() { return Uint8Array.from(sourceBytes); },
    select(drawIndex) {
      if (!Number.isInteger(drawIndex) || drawIndex < 0 || drawIndex >= count) {
        throw new GeneratedDatasetsArtifactError(
          `generated dataset index ${String(drawIndex)} is outside range 0..${count - 1}`,
        );
      }
      const selection = selections[drawIndex];
      return Object.freeze({
        drawIndex,
        parametersBytes: Uint8Array.from(selection.parametersBytes),
        datasetBytes: Uint8Array.from(selection.datasetBytes),
      });
    },
  };
  Object.defineProperty(artifact, "_source", { value: Object.freeze(source) });
  return Object.freeze(artifact);
}

/**
 * @param {ReturnType<typeof parseGeneratedDatasets>} artifact
 * @param {{modelBytes: Uint8Array, designBytes: Uint8Array, fixedParametersBytes?: Uint8Array}} sources
 */
export async function verifyGeneratedDatasets(artifact, sources) {
  if (
    sources.expectedSourceKind !== undefined &&
    artifact.sourceKind !== sources.expectedSourceKind
  ) {
    throw new GeneratedDatasetsArtifactError(
      `generated source kind ${artifact.sourceKind} does not match requested source kind ${sources.expectedSourceKind}`,
    );
  }
  if (sources.expectedCount !== undefined && artifact.count !== sources.expectedCount) {
    throw new GeneratedDatasetsArtifactError(
      `generated count ${artifact.count} does not match requested count ${sources.expectedCount}`,
    );
  }
  if (sources.expectedSeed !== undefined && artifact.seed !== sources.expectedSeed) {
    throw new GeneratedDatasetsArtifactError(
      `generated seed ${artifact.seed} does not match requested seed ${sources.expectedSeed}`,
    );
  }
  if (await sha256(sources.modelBytes) !== artifact.generationModelHash) {
    throw new GeneratedDatasetsArtifactError("generation model hash does not match resolved bytes");
  }
  if (await sha256(sources.designBytes) !== artifact.designHash) {
    throw new GeneratedDatasetsArtifactError("design hash does not match resolved bytes");
  }
  const design = parseCanonicalBytes(sources.designBytes, "design");
  const designEntries = Object.entries(design.variables);
  for (const draw of artifact.draws) {
    const prefix = Object.entries(draw.dataset.variables).slice(0, designEntries.length);
    if (JSON.stringify(prefix) !== JSON.stringify(designEntries)) {
      throw new GeneratedDatasetsArtifactError(
        `draw ${draw.drawIndex} dataset does not preserve the resolved design prefix`,
      );
    }
  }
  if (artifact.sourceKind === "fixed") {
    if (sources.fixedParametersBytes === undefined) {
      throw new GeneratedDatasetsArtifactError("fixed parameters bytes are required for verification");
    }
    if (await sha256(sources.fixedParametersBytes) !== artifact._source.sourceHash) {
      throw new GeneratedDatasetsArtifactError("fixed parameters hash does not match resolved bytes");
    }
    const fixed = parseCanonicalBytes(sources.fixedParametersBytes, "fixed parameters");
    for (const draw of artifact.draws) {
      if (JSON.stringify(draw.parameters) !== JSON.stringify(fixed)) {
        throw new GeneratedDatasetsArtifactError(
          `draw ${draw.drawIndex} parameters differ from fixed parameters`,
        );
      }
    }
    return;
  }
  if (artifact.sourceKind === "model-prior") {
    if (sources.modelPriorBytes === undefined) {
      throw new GeneratedDatasetsArtifactError(
        "model-prior model bytes are required for verification",
      );
    }
    const provenance = sources.authoredProvenance ?? null;
    if (
      await sha256(sources.modelPriorBytes) !== artifact._source.modelHash ||
      JSON.stringify(provenance) !== JSON.stringify(artifact._source.authoredProvenance)
    ) {
      throw new GeneratedDatasetsArtifactError(
        "model-prior descriptor does not match the requested source",
      );
    }
    return;
  }
  if (sources.posteriorBytes === undefined || sources.fitDataBytes === undefined) {
    throw new GeneratedDatasetsArtifactError(
      "posterior and fit-data bytes are required for verification",
    );
  }
  if (
    await sha256(sources.posteriorBytes) !== artifact._source.fitHash ||
    await sha256(sources.modelBytes) !== artifact._source.fitModelHash ||
    await sha256(sources.fitDataBytes) !== artifact._source.fitDataHash
  ) {
    throw new GeneratedDatasetsArtifactError(
      "posterior descriptor does not match the requested source",
    );
  }
  let posterior;
  try {
    posterior = await validatePortablePosterior({
      modelBytes: sources.modelBytes,
      dataBytes: sources.fitDataBytes,
      posteriorBytes: sources.posteriorBytes,
      requireFingerprint: sources.posteriorAssociation !== "runtime",
    });
  } catch (error) {
    if (error instanceof PortablePosteriorError) {
      throw new GeneratedDatasetsArtifactError(error.message);
    }
    throw error;
  }
  const sourceShapes = posterior.parameters.map((parameter) => [
    parameter.name, parameter.shape,
  ]);
  for (const draw of artifact.draws) {
    const lineage = draw.sourceLineage;
    const sourceDraw = posterior.draws[lineage.source_draw_index];
    if (sourceDraw === undefined || sourceDraw.chain !== lineage.chain ||
        sourceDraw.draw !== lineage.draw) {
      throw new GeneratedDatasetsArtifactError(
        `draw ${draw.drawIndex} posterior lineage does not match source`,
      );
    }
    const actualShapes = Object.entries(draw.parameters.variables).map(([name, value]) => [
      name, value.shape,
    ]);
    const actualValues = Object.entries(draw.parameters.variables).map(([name, value]) => [
      name, value.values,
    ]);
    if (JSON.stringify(actualShapes) !== JSON.stringify(sourceShapes) ||
        JSON.stringify(actualValues) !== JSON.stringify(sourceDraw.values)) {
      throw new GeneratedDatasetsArtifactError(
        `draw ${draw.drawIndex} parameters differ from posterior source`,
      );
    }
  }
}

function splitLines(bytes) {
  const lines = [];
  let start = 0;
  for (let index = 0; index < bytes.byteLength; index += 1) {
    if (bytes[index] === 0x0a) {
      lines.push(bytes.slice(start, index));
      start = index + 1;
    }
  }
  return lines;
}

function parseLine(line, index) {
  let text;
  try {
    text = UTF8.decode(line);
  } catch {
    throw new GeneratedDatasetsArtifactError(`generated-dataset line ${index} is not UTF-8`);
  }
  try {
    assertUniqueObjectKeys(text, `generated-dataset line ${index}`);
    return JSON.parse(text);
  } catch (error) {
    throw new GeneratedDatasetsArtifactError(
      `generated-dataset line ${index} is not finite valid JSON: ${String(error)}`,
    );
  }
}

function assertUniqueObjectKeys(text, label) {
  let index = 0;
  const whitespace = () => {
    while (/\s/u.test(text[index] ?? "")) index += 1;
  };
  const string = () => {
    const start = index;
    index += 1;
    let escaped = false;
    while (index < text.length) {
      const character = text[index++];
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') return JSON.parse(text.slice(start, index));
    }
    throw new GeneratedDatasetsArtifactError(`${label} has malformed string`);
  };
  const value = () => {
    whitespace();
    if (text[index] === "{") {
      objectValue();
    } else if (text[index] === "[") {
      index += 1;
      whitespace();
      if (text[index] === "]") { index += 1; return; }
      while (true) {
        value();
        whitespace();
        if (text[index] === "]") { index += 1; return; }
        index += 1;
      }
    } else if (text[index] === '"') {
      string();
    } else {
      while (index < text.length && !",]}".includes(text[index])) index += 1;
    }
  };
  const objectValue = () => {
    index += 1;
    const keys = new Set();
    whitespace();
    if (text[index] === "}") { index += 1; return; }
    while (true) {
      whitespace();
      const key = string();
      if (keys.has(key)) {
        throw new GeneratedDatasetsArtifactError(`${label} has duplicate object key ${key}`);
      }
      keys.add(key);
      whitespace();
      index += 1;
      value();
      whitespace();
      if (text[index] === "}") { index += 1; return; }
      index += 1;
    }
  };
  value();
}

function object(value, label) {
  if (!isObject(value)) {
    throw new GeneratedDatasetsArtifactError(`${label} must be a JSON object`);
  }
  return value;
}

function exactKeys(value, expected, label) {
  const actual = Object.keys(value);
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new GeneratedDatasetsArtifactError(
      `${label} has unknown, missing, or out-of-order keys: ${JSON.stringify(actual)}`,
    );
  }
}

function marker(value, label) {
  if (MARKER_KEYS.some((key, index) => value[key] !== MARKER_VALUES[index])) {
    throw new GeneratedDatasetsArtifactError(`${label} generated-dataset marker is invalid`);
  }
}

function phases(value, label) {
  if (JSON.stringify(value) !== JSON.stringify(PHASES)) {
    throw new GeneratedDatasetsArtifactError(`${label} workflow phases are invalid`);
  }
}

function hash(value, label) {
  if (typeof value !== "string" || !HASH.test(value)) {
    throw new GeneratedDatasetsArtifactError(`${label} must be a lowercase sha256 hash`);
  }
  return value;
}

function parameterSource(value) {
  const source = object(value, "parameter_source");
  if (source.kind === "fixed") {
    exactKeys(source, ["kind", "parameters_hash"], "fixed parameter_source");
    return { kind: source.kind, sourceHash: hash(source.parameters_hash, "parameters_hash") };
  }
  if (source.kind === "model-prior") {
    exactKeys(source, ["kind", "model_hash", "authored_provenance"], "model-prior parameter_source");
    hash(source.model_hash, "model_hash");
    if (source.authored_provenance !== null) {
      const claims = object(source.authored_provenance, "authored_provenance");
      exactKeys(claims, ["claimed_source_model_hash", "claimed_outcome_model_hash"], "authored_provenance");
      hash(claims.claimed_source_model_hash, "claimed_source_model_hash");
      hash(claims.claimed_outcome_model_hash, "claimed_outcome_model_hash");
    }
    return {
      kind: source.kind,
      sourceHash: null,
      modelHash: source.model_hash,
      authoredProvenance: source.authored_provenance === null
        ? null
        : Object.freeze({ ...source.authored_provenance }),
    };
  }
  if (source.kind === "posterior") {
    exactKeys(source, ["kind", "fit_hash", "fit_model_hash", "fit_data_hash"], "posterior parameter_source");
    for (const name of ["fit_hash", "fit_model_hash", "fit_data_hash"]) hash(source[name], name);
    return {
      kind: source.kind,
      sourceHash: null,
      fitHash: source.fit_hash,
      fitModelHash: source.fit_model_hash,
      fitDataHash: source.fit_data_hash,
    };
  }
  throw new GeneratedDatasetsArtifactError(`parameter_source has unknown kind ${String(source.kind)}`);
}

function schema(value, label) {
  if (!Array.isArray(value)) throw new GeneratedDatasetsArtifactError(`${label} must be an array`);
  const names = new Set();
  return value.map((raw, index) => {
    const entry = object(raw, `${label}[${index}]`);
    exactKeys(entry, ["name", "dtype", "shape"], `${label}[${index}]`);
    if (typeof entry.name !== "string" || entry.name.length === 0 || names.has(entry.name)) {
      throw new GeneratedDatasetsArtifactError(`${label}[${index}] has invalid or duplicate name`);
    }
    if (!["bool", "int32", "int64", "float32", "float64"].includes(entry.dtype)) {
      throw new GeneratedDatasetsArtifactError(`${label}[${index}] has invalid dtype`);
    }
    if (!Array.isArray(entry.shape)) {
      throw new GeneratedDatasetsArtifactError(`${label}[${index}] shape must be an array`);
    }
    const shape = entry.shape.map((dim) => integer(dim, `${label}[${index}] shape`, 0, MAX_SAFE_INTEGER));
    names.add(entry.name);
    return Object.freeze({ name: entry.name, dtype: entry.dtype, shape: Object.freeze(shape) });
  });
}

function canonicalDocument(value, label) {
  if (!isObject(value) || value.format !== "bayescycle.data.json.v1") {
    throw new GeneratedDatasetsArtifactError(`${label} must be a strict canonical data document`);
  }
  try {
    return deepFreeze(normalizeDocument(value));
  } catch (error) {
    throw new GeneratedDatasetsArtifactError(`${label}: ${String(error)}`);
  }
}

function parseCanonicalBytes(bytes, label) {
  let value;
  try {
    value = JSON.parse(UTF8.decode(bytes));
  } catch (error) {
    throw new GeneratedDatasetsArtifactError(`${label} is not finite valid JSON: ${String(error)}`);
  }
  return canonicalDocument(value, label);
}

function matchesSchema(document, expected, label) {
  const actual = Object.entries(document.variables).map(([name, variable]) => ({
    name,
    dtype: variable.dtype,
    shape: variable.shape,
  }));
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new GeneratedDatasetsArtifactError(`${label} does not match declared schema`);
  }
}

function lineage(value, sourceKind, drawIndex) {
  const source = object(value, `draw ${drawIndex} source_lineage`);
  if (sourceKind === "fixed") {
    exactKeys(source, ["kind"], `draw ${drawIndex} fixed source_lineage`);
  } else if (sourceKind === "model-prior") {
    exactKeys(source, ["kind", "source_draw_index"], `draw ${drawIndex} model-prior source_lineage`);
    if (source.source_draw_index !== drawIndex) {
      throw new GeneratedDatasetsArtifactError(
        `draw ${drawIndex} model-prior source_draw_index must equal draw_index`,
      );
    }
  } else {
    exactKeys(source, ["kind", "source_draw_index", "chain", "draw"], `draw ${drawIndex} posterior source_lineage`);
    for (const name of ["source_draw_index", "chain", "draw"]) {
      integer(source[name], name, 0, MAX_SAFE_INTEGER);
    }
  }
  if (source.kind !== sourceKind) {
    throw new GeneratedDatasetsArtifactError(
      `draw ${drawIndex} source lineage kind disagrees with parameter source`,
    );
  }
  return Object.freeze({ ...source });
}

function integer(value, label, minimum, maximum) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new GeneratedDatasetsArtifactError(
      `${label} must be an integer in range ${minimum}..${maximum}`,
    );
  }
  return value;
}

async function sha256(bytes) {
  if (!(bytes instanceof Uint8Array)) {
    throw new GeneratedDatasetsArtifactError("hash source must be bytes");
  }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", Uint8Array.from(bytes)));
  return `sha256:${[...digest].map((value) => value.toString(16).padStart(2, "0")).join("")}`;
}

function validateDepth(line, lineIndex) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (const byte of line) {
    if (inString) {
      if (escaped) escaped = false;
      else if (byte === 0x5c) escaped = true;
      else if (byte === 0x22) inString = false;
    } else if (byte === 0x22) inString = true;
    else if (byte === 0x7b || byte === 0x5b) {
      depth += 1;
      if (depth > MAX_DEPTH) {
        throw new GeneratedDatasetsArtifactError(`generated-dataset line ${lineIndex} exceeds nesting depth`);
      }
    } else if (byte === 0x7d || byte === 0x5d) {
      depth -= 1;
      if (depth < 0) {
        throw new GeneratedDatasetsArtifactError(`generated-dataset line ${lineIndex} has malformed nesting`);
      }
    }
  }
  if (inString || depth !== 0) {
    throw new GeneratedDatasetsArtifactError(`generated-dataset line ${lineIndex} has malformed finite JSON`);
  }
}

function topLevelValueSpan(line, key, offset) {
  const markerBytes = new TextEncoder().encode(`"${key}"`);
  const keyStart = findBytes(line, markerBytes, offset);
  if (keyStart < 0) throw new GeneratedDatasetsArtifactError(`draw is missing raw ${key} bytes`);
  let colon = keyStart + markerBytes.length;
  while (colon < line.length && line[colon] !== 0x3a) colon += 1;
  let start = colon + 1;
  while ([0x20, 0x09, 0x0d].includes(line[start])) start += 1;
  if (![0x7b, 0x5b].includes(line[start])) {
    throw new GeneratedDatasetsArtifactError(`draw ${key} must be a JSON container`);
  }
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let position = start; position < line.length; position += 1) {
    const byte = line[position];
    if (inString) {
      if (escaped) escaped = false;
      else if (byte === 0x5c) escaped = true;
      else if (byte === 0x22) inString = false;
      continue;
    }
    if (byte === 0x22) inString = true;
    else if (byte === 0x7b || byte === 0x5b) depth += 1;
    else if (byte === 0x7d || byte === 0x5d) {
      depth -= 1;
      if (depth === 0) return { bytes: line.slice(start, position + 1), end: position + 1 };
    }
  }
  throw new GeneratedDatasetsArtifactError(`draw ${key} has no closing delimiter`);
}

function findBytes(haystack, needle, offset) {
  outer: for (let index = offset; index <= haystack.length - needle.length; index += 1) {
    for (let part = 0; part < needle.length; part += 1) {
      if (haystack[index + part] !== needle[part]) continue outer;
    }
    return index;
  }
  return -1;
}

function withLf(bytes) {
  const result = new Uint8Array(bytes.length + 1);
  result.set(bytes);
  result[result.length - 1] = 0x0a;
  return result;
}

function deepFreeze(value) {
  if (Array.isArray(value)) {
    for (const item of value) deepFreeze(item);
  } else if (isObject(value)) {
    for (const item of Object.values(value)) deepFreeze(item);
  }
  return Object.freeze(value);
}

function isObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
