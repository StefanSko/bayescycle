const FORMAT = "bayescycle.data.json.v1";
const DTYPES = new Set(["bool", "int32", "int64", "float32", "float64"]);
const INT32_MIN = -(2 ** 31);
const INT32_MAX = 2 ** 31 - 1;

/** @param {unknown} value */
export function normalizeDocument(value) {
  if (!isObject(value)) throw new Error("data document must be a JSON object");
  if (value.format === FORMAT) return parseCanonical(value);
  return {
    format: FORMAT,
    variables: Object.fromEntries(
      Object.entries(value).map(([name, entry]) => [name, normalizePlainVariable(entry, name)]),
    ),
  };
}

/** @param {string} text */
export function parseDocument(text) {
  let value;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  return normalizeDocument(value);
}

/** @param {ReturnType<typeof normalizeDocument>} document */
export function serializeDocument(document) {
  return `${JSON.stringify(document)}\n`;
}

function parseCanonical(value) {
  const keys = Object.keys(value);
  if (keys.length !== 2 || !keys.includes("format") || !keys.includes("variables")) {
    throw new Error("canonical data document requires only format and variables");
  }
  if (!isObject(value.variables)) throw new Error("canonical variables must be an object");
  return {
    format: FORMAT,
    variables: Object.fromEntries(
      Object.entries(value.variables).map(([name, entry]) => [name, parseCanonicalVariable(entry, name)]),
    ),
  };
}

function parseCanonicalVariable(value, name) {
  if (!isObject(value)) throw new Error(`variable ${name} must be an object`);
  const keys = Object.keys(value);
  if (keys.length !== 3 || !keys.includes("dtype") || !keys.includes("shape") || !keys.includes("values")) {
    throw new Error(`variable ${name} requires only dtype, shape, and values`);
  }
  if (typeof value.dtype !== "string" || !DTYPES.has(value.dtype)) {
    throw new Error(`variable ${name} has unsupported dtype`);
  }
  if (!Array.isArray(value.shape) || !value.shape.every(validDimension)) {
    throw new Error(`variable ${name} has invalid shape`);
  }
  if (!Array.isArray(value.values)) throw new Error(`variable ${name} values must be an array`);
  const expected = value.shape.length === 0 ? 1 : value.shape.reduce((a, b) => a * b, 1);
  if (value.values.length !== expected) {
    throw new Error(`variable ${name} shape expects ${expected} values, got ${value.values.length}`);
  }
  const values = value.values.map((entry) => canonicalScalar(entry, value.dtype, name));
  return { dtype: value.dtype, shape: [...value.shape], values };
}

function normalizePlainVariable(value, name) {
  const { shape, values } = flatten(value, name);
  const dtype = inferDtype(values, name);
  return { dtype, shape, values: values.map((entry) => canonicalScalar(entry, dtype, name)) };
}

function flatten(value, name) {
  if (!Array.isArray(value)) {
    if (!validPlainScalar(value)) throw new Error(`variable ${name} must contain booleans or numbers`);
    return { shape: [], values: [value] };
  }
  if (value.length === 0) return { shape: [0], values: [] };
  const children = value.map((entry) => flatten(entry, name));
  const childShape = children[0].shape;
  if (!children.every((entry) => sameShape(entry.shape, childShape))) {
    throw new Error(`variable ${name} arrays must be rectangular`);
  }
  return { shape: [value.length, ...childShape], values: children.flatMap((entry) => entry.values) };
}

function inferDtype(values, name) {
  if (values.length === 0) return "float64";
  if (values.every((value) => typeof value === "boolean")) return "bool";
  if (values.some((value) => typeof value === "boolean")) {
    throw new Error(`variable ${name} must not mix boolean and numeric values`);
  }
  return values.every((value) => Number.isInteger(value)) ? "int64" : "float64";
}

function canonicalScalar(value, dtype, name) {
  if (dtype === "bool") {
    if (typeof value !== "boolean") throw new Error(`variable ${name} bool values must be boolean`);
    return value;
  }
  if (dtype === "int32" || dtype === "int64") {
    if (!Number.isSafeInteger(value)) throw new Error(`variable ${name} integer values must be safe JSON integers`);
    if (dtype === "int32" && (value < INT32_MIN || value > INT32_MAX)) {
      throw new Error(`variable ${name} value is outside int32 range`);
    }
    return value;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`variable ${name} floating values must be finite numbers`);
  }
  return value;
}

function validPlainScalar(value) {
  return typeof value === "boolean" || (typeof value === "number" && Number.isFinite(value));
}

function validDimension(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function sameShape(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function isObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
