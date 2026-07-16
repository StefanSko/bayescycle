import { parseStrictJson } from "./strict-json.mjs";

export const MAX_GENERATION_COUNT = 1000;
export const MAX_GENERATION_INPUT_BYTES = 8 * 1024 * 1024;
export const MAX_GENERATION_PLAN_BYTES = 1024 * 1024;

const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER;
const MAX_DEPTH = 64;
const HASH = /^sha256:[0-9a-f]{64}$/u;
const UTF8 = new TextEncoder();
const TEXT = new TextDecoder("utf-8", { fatal: true });

export class GenerationPlanError extends Error {
  constructor(message) {
    super(message);
    this.name = "GenerationPlanError";
  }
}

export function fixed(parametersBytes) {
  const owned = copyBytes(parametersBytes, "fixed parameters");
  return Object.freeze({
    kind: "fixed",
    get parametersBytes() { return Uint8Array.from(owned); },
  });
}

export function modelPrior(modelIrBytes, authoredProvenance = null) {
  const owned = copyBytes(modelIrBytes, "model-prior model IR");
  const provenance = authoredProvenanceValue(authoredProvenance);
  return Object.freeze({
    kind: "model-prior",
    get modelIrBytes() { return Uint8Array.from(owned); },
    authoredProvenance: provenance,
  });
}

export function fitArtifact(modelIrBytes, dataBytes, posteriorBytes, association) {
  const model = copyBytes(modelIrBytes, "fit model IR");
  const data = copyBytes(dataBytes, "fit data");
  const posterior = copyBytes(posteriorBytes, "fit posterior");
  if (association !== "runtime" && association !== "portable") {
    throw new GenerationPlanError("fit association must be runtime or portable");
  }
  return Object.freeze({
    kind: "fit-artifact",
    get modelIrBytes() { return Uint8Array.from(model); },
    get dataBytes() { return Uint8Array.from(data); },
    get posteriorBytes() { return Uint8Array.from(posterior); },
    association,
  });
}

export function posteriorOf(value) {
  validateFitArtifact(value);
  return Object.freeze({ kind: "posterior", fitArtifact: value });
}

export function outcomesOf(modelIrBytes, designBytes) {
  const model = copyBytes(modelIrBytes, "outcomes model IR");
  const design = copyBytes(designBytes, "generation design");
  return Object.freeze({
    kind: "model-outcomes",
    get modelIrBytes() { return Uint8Array.from(model); },
    get designBytes() { return Uint8Array.from(design); },
  });
}

export function jointPredict(parameters, outcomes) {
  validateParameterSource(parameters);
  validateOutcomes(outcomes);
  const sourceModel = sourceModelBytes(parameters);
  if (sourceModel !== null && !equalBytes(sourceModel, outcomes.modelIrBytes)) {
    throw new GenerationPlanError("parameter source model must equal the outcomes model bytes");
  }
  return Object.freeze({ kind: "joint-predict", parameters, outcomes });
}

export function draw(distribution, count, seed, designSource = undefined) {
  validateJointPredict(distribution);
  integer(count, "count", 1, MAX_GENERATION_COUNT);
  integer(seed, "seed", 0, MAX_SAFE_INTEGER);
  if (designSource === undefined) {
    return Object.freeze({ kind: "draw", count, seed, distribution });
  }
  return Object.freeze({
    kind: "draw",
    count,
    seed,
    designSource: designSourceValue(designSource),
    distribution,
  });
}

export function generateDatasets(modelIrBytes, options) {
  if (!isObject(options)) throw new GenerationPlanError("generation options must be an object");
  const allowed = new Set(["design", "designSource", "parameterSource", "count", "seed"]);
  const unknown = Object.keys(options).filter((key) => !allowed.has(key));
  if (unknown.length > 0 || !("design" in options) || !("parameterSource" in options)) {
    throw new GenerationPlanError(`generation options have unknown or missing fields: ${JSON.stringify(unknown)}`);
  }
  return draw(
    jointPredict(options.parameterSource, outcomesOf(modelIrBytes, options.design)),
    options.count ?? 100,
    options.seed ?? 0,
    options.designSource,
  );
}

export function validateGenerationPlan(value) {
  const hasDesignSource = isObject(value) && Object.hasOwn(value, "designSource");
  exactKeys(
    value,
    hasDesignSource
      ? ["kind", "count", "seed", "designSource", "distribution"]
      : ["kind", "count", "seed", "distribution"],
    "draw plan",
  );
  if (value.kind !== "draw") throw new GenerationPlanError("plan kind must be draw");
  integer(value.count, "count", 1, MAX_GENERATION_COUNT);
  integer(value.seed, "seed", 0, MAX_SAFE_INTEGER);
  if (hasDesignSource) designSourceValue(value.designSource);
  validateJointPredict(value.distribution);
  return value;
}

export async function serializeGenerationPlan(plan) {
  validateGenerationPlan(plan);
  const parameters = await sourceDocument(plan.distribution.parameters);
  const outcomes = plan.distribution.outcomes;
  const prefix = {
    generation_plan_format: "v0-provisional",
    kind: "draw",
    count: plan.count,
    seed: plan.seed,
  };
  const distribution = {
    kind: "joint-predict",
    parameters,
    outcomes: {
      kind: "model-outcomes",
      model_hash: await sha256(outcomes.modelIrBytes),
      design_hash: await sha256(outcomes.designBytes),
    },
  };
  const text = Object.hasOwn(plan, "designSource")
    ? `${JSON.stringify(prefix).slice(0, -1)},"design_source":${serializeDesignSource(plan.designSource)},"distribution":${JSON.stringify(distribution)}}\n`
    : `${JSON.stringify({ ...prefix, distribution })}\n`;
  const bytes = UTF8.encode(text);
  if (bytes.byteLength > MAX_GENERATION_PLAN_BYTES) {
    throw new GenerationPlanError("serialized generation plan exceeds byte limit");
  }
  return bytes;
}

export async function parseGenerationPlanDocument(input) {
  const bytes = copyBytes(input, "generation plan", MAX_GENERATION_PLAN_BYTES);
  if (bytes.at(-1) !== 0x0a) throw new GenerationPlanError("generation plan must end in one LF");
  validateDepth(bytes);
  let value;
  try {
    value = parseStrictJson(TEXT.decode(bytes), "generation plan", {
      integerKeys: ["count", "seed"],
      // Provenance keys are user-named design slots; structural field rules
      // (integer tokens, dtype/values coupling) must not apply inside it.
      unrestrictedObjectKeys: ["design_source"],
      sortedObjectKeys: ["design_source"],
    });
  }
  catch (error) { throw new GenerationPlanError(`generation plan is not valid JSON: ${String(error)}`); }
  if (!isObject(value) || value.generation_plan_format !== "v0-provisional") {
    throw new GenerationPlanError("generation plan format must be v0-provisional");
  }
  const hasDesignSource = Object.hasOwn(value, "design_source");
  exactKeys(
    value,
    hasDesignSource
      ? ["generation_plan_format", "kind", "count", "seed", "design_source", "distribution"]
      : ["generation_plan_format", "kind", "count", "seed", "distribution"],
    "generation plan",
  );
  if (value.kind !== "draw") throw new GenerationPlanError("generation plan kind must be draw");
  integer(value.count, "count", 1, MAX_GENERATION_COUNT);
  integer(value.seed, "seed", 0, MAX_SAFE_INTEGER);
  if (hasDesignSource) designSourceValue(value.design_source);
  const distribution = value.distribution;
  exactKeys(distribution, ["kind", "parameters", "outcomes"], "distribution");
  if (distribution.kind !== "joint-predict") {
    throw new GenerationPlanError("distribution kind must be joint-predict");
  }
  const sourceModelHash = parseSourceDocument(distribution.parameters);
  const outcomes = distribution.outcomes;
  exactKeys(outcomes, ["kind", "model_hash", "design_hash"], "outcomes");
  if (outcomes.kind !== "model-outcomes") {
    throw new GenerationPlanError("outcomes kind must be model-outcomes");
  }
  hash(outcomes.model_hash, "model_hash");
  hash(outcomes.design_hash, "design_hash");
  if (sourceModelHash !== null && sourceModelHash !== outcomes.model_hash) {
    throw new GenerationPlanError("parameter source model hash must equal outcomes model hash");
  }
  const identityHash = await sha256(bytes);
  const invalidationKey = await sha256(concat(UTF8.encode("bayescycle-generation-key-v0\n"), bytes));
  return Object.freeze({
    get bytes() { return Uint8Array.from(bytes); },
    identityHash,
    invalidationKey,
  });
}

export async function generationPlanIdentity(plan) {
  return sha256(await serializeGenerationPlan(plan));
}

export async function generationInvalidationKey(plan) {
  return sha256(concat(
    UTF8.encode("bayescycle-generation-key-v0\n"),
    await serializeGenerationPlan(plan),
  ));
}

function validateParameterSource(source) {
  if (!isObject(source)) throw new GenerationPlanError("parameter source must be an object");
  if (source.kind === "fixed") {
    exactKeys(source, ["kind", "parametersBytes"], "fixed source");
    copyBytes(source.parametersBytes, "fixed parameters");
    return;
  }
  if (source.kind === "model-prior") {
    exactKeys(source, ["kind", "modelIrBytes", "authoredProvenance"], "model-prior source");
    copyBytes(source.modelIrBytes, "model-prior model IR");
    authoredProvenanceValue(source.authoredProvenance);
    return;
  }
  if (source.kind === "posterior") {
    exactKeys(source, ["kind", "fitArtifact"], "posterior source");
    validateFitArtifact(source.fitArtifact);
    return;
  }
  throw new GenerationPlanError(`parameter source has unknown kind ${String(source.kind)}`);
}

function validateFitArtifact(value) {
  exactKeys(
    value,
    ["kind", "modelIrBytes", "dataBytes", "posteriorBytes", "association"],
    "fit artifact",
  );
  if (value.kind !== "fit-artifact") throw new GenerationPlanError("fit artifact kind is invalid");
  copyBytes(value.modelIrBytes, "fit model IR");
  copyBytes(value.dataBytes, "fit data");
  copyBytes(value.posteriorBytes, "fit posterior");
  if (value.association !== "runtime" && value.association !== "portable") {
    throw new GenerationPlanError("fit association must be runtime or portable");
  }
}

function validateOutcomes(value) {
  exactKeys(value, ["kind", "modelIrBytes", "designBytes"], "outcomes");
  if (value.kind !== "model-outcomes") throw new GenerationPlanError("outcomes kind is invalid");
  copyBytes(value.modelIrBytes, "outcomes model IR");
  copyBytes(value.designBytes, "generation design");
}

function validateJointPredict(value) {
  exactKeys(value, ["kind", "parameters", "outcomes"], "joint prediction");
  if (value.kind !== "joint-predict") throw new GenerationPlanError("distribution kind is invalid");
  validateParameterSource(value.parameters);
  validateOutcomes(value.outcomes);
  const sourceModel = sourceModelBytes(value.parameters);
  if (sourceModel !== null && !equalBytes(sourceModel, value.outcomes.modelIrBytes)) {
    throw new GenerationPlanError("parameter source model must equal the outcomes model bytes");
  }
}

function sourceModelBytes(source) {
  if (source.kind === "model-prior") return source.modelIrBytes;
  if (source.kind === "posterior") return source.fitArtifact.modelIrBytes;
  return null;
}

async function sourceDocument(source) {
  if (source.kind === "fixed") {
    return { kind: "fixed", parameters_hash: await sha256(source.parametersBytes) };
  }
  if (source.kind === "model-prior") {
    const provenance = source.authoredProvenance === null ? null : {
      claimed_source_model_hash: source.authoredProvenance.claimedSourceModelHash,
      claimed_outcome_model_hash: source.authoredProvenance.claimedOutcomeModelHash,
    };
    return {
      kind: "model-prior",
      model_hash: await sha256(source.modelIrBytes),
      authored_provenance: provenance,
    };
  }
  return {
    kind: "posterior",
    fit_hash: await sha256(source.fitArtifact.posteriorBytes),
    fit_model_hash: await sha256(source.fitArtifact.modelIrBytes),
    fit_data_hash: await sha256(source.fitArtifact.dataBytes),
  };
}

function parseSourceDocument(source) {
  exactKeys(source, Object.keys(source), "parameter source");
  if (source.kind === "fixed") {
    exactKeys(source, ["kind", "parameters_hash"], "fixed parameter source");
    hash(source.parameters_hash, "parameters_hash");
    return null;
  }
  if (source.kind === "model-prior") {
    exactKeys(source, ["kind", "model_hash", "authored_provenance"], "model-prior parameter source");
    hash(source.model_hash, "model_hash");
    if (source.authored_provenance !== null) {
      exactKeys(source.authored_provenance, ["claimed_source_model_hash", "claimed_outcome_model_hash"], "authored provenance");
      hash(source.authored_provenance.claimed_source_model_hash, "claimed_source_model_hash");
      hash(source.authored_provenance.claimed_outcome_model_hash, "claimed_outcome_model_hash");
    }
    return source.model_hash;
  }
  if (source.kind === "posterior") {
    exactKeys(source, ["kind", "fit_hash", "fit_model_hash", "fit_data_hash"], "posterior parameter source");
    for (const name of ["fit_hash", "fit_model_hash", "fit_data_hash"]) hash(source[name], name);
    return source.fit_model_hash;
  }
  throw new GenerationPlanError(`parameter source has unknown kind ${String(source.kind)}`);
}

function authoredProvenanceValue(value) {
  if (value === null) return null;
  exactKeys(value, ["claimedSourceModelHash", "claimedOutcomeModelHash"], "authored provenance");
  hash(value.claimedSourceModelHash, "claimedSourceModelHash");
  hash(value.claimedOutcomeModelHash, "claimedOutcomeModelHash");
  return Object.freeze({
    claimedSourceModelHash: value.claimedSourceModelHash,
    claimedOutcomeModelHash: value.claimedOutcomeModelHash,
  });
}

function designSourceValue(value) {
  if (!isObject(value)) throw new GenerationPlanError("design_source must be an object");
  const entries = Object.entries(value);
  for (const [name, expression] of entries) {
    if (name.length === 0) {
      throw new GenerationPlanError("design_source keys must be non-empty strings");
    }
    if (!name.isWellFormed()) {
      throw new GenerationPlanError(
        "design_source keys must be well-formed Unicode scalar-value strings",
      );
    }
    if (typeof expression !== "string" || expression.length === 0) {
      throw new GenerationPlanError("design_source values must be non-empty strings");
    }
    if (!expression.isWellFormed()) {
      throw new GenerationPlanError(
        "design_source values must be well-formed Unicode scalar-value strings",
      );
    }
  }
  return Object.freeze(Object.fromEntries(entries));
}

function serializeDesignSource(value) {
  const entries = Object.entries(value).sort(
    ([left], [right]) => compareCodePointStrings(left, right),
  );
  return `{${entries.map(([name, expression]) =>
    `${JSON.stringify(name)}:${JSON.stringify(expression)}`).join(",")}}`;
}

function compareCodePointStrings(left, right) {
  const leftPoints = Array.from(left, (character) => character.codePointAt(0));
  const rightPoints = Array.from(right, (character) => character.codePointAt(0));
  const length = Math.min(leftPoints.length, rightPoints.length);
  for (let index = 0; index < length; index += 1) {
    if (leftPoints[index] !== rightPoints[index]) {
      return leftPoints[index] - rightPoints[index];
    }
  }
  return leftPoints.length - rightPoints.length;
}

function copyBytes(value, label, maximum = MAX_GENERATION_INPUT_BYTES) {
  if (!(value instanceof Uint8Array)) throw new GenerationPlanError(`${label} must be bytes`);
  if (value.byteLength === 0) throw new GenerationPlanError(`${label} bytes must not be empty`);
  if (value.byteLength > maximum) throw new GenerationPlanError(`${label} exceeds ${maximum} bytes`);
  return Uint8Array.from(value);
}

function validateDepth(bytes) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (const byte of bytes) {
    if (inString) {
      if (escaped) escaped = false;
      else if (byte === 0x5c) escaped = true;
      else if (byte === 0x22) inString = false;
    } else if (byte === 0x22) inString = true;
    else if (byte === 0x7b || byte === 0x5b) {
      depth += 1;
      if (depth > MAX_DEPTH) {
        throw new GenerationPlanError(`generation plan exceeds nesting depth ${MAX_DEPTH}`);
      }
    } else if (byte === 0x7d || byte === 0x5d) {
      depth -= 1;
      if (depth < 0) throw new GenerationPlanError("generation plan has malformed nesting");
    }
  }
  if (inString || depth !== 0) {
    throw new GenerationPlanError("generation plan has malformed nesting");
  }
}

function exactKeys(value, expected, label) {
  if (!isObject(value)) throw new GenerationPlanError(`${label} must be an object`);
  const actual = Object.keys(value);
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new GenerationPlanError(
      `${label} has unknown, missing, or out-of-order fields: ${JSON.stringify(actual)}`,
    );
  }
}

function integer(value, label, minimum, maximum) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new GenerationPlanError(`${label} must be an integer in ${minimum}..${maximum}`);
  }
  return value;
}

function hash(value, label) {
  if (typeof value !== "string" || !HASH.test(value)) {
    throw new GenerationPlanError(`${label} must be a lowercase sha256 hash`);
  }
  return value;
}

async function sha256(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", Uint8Array.from(bytes)));
  return `sha256:${[...digest].map((value) => value.toString(16).padStart(2, "0")).join("")}`;
}

function concat(left, right) {
  const bytes = new Uint8Array(left.byteLength + right.byteLength);
  bytes.set(left);
  bytes.set(right, left.byteLength);
  return bytes;
}

function equalBytes(left, right) {
  return left.byteLength === right.byteLength && left.every((value, index) => value === right[index]);
}

function isObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
