const FORMAT = "bayescycle.data.json.v1";
const DTYPES = new Set(["bool", "int32", "int64", "float32", "float64"]);
const INT32_MIN = -(2 ** 31);
const INT32_MAX = 2 ** 31 - 1;
export const MAX_DOCUMENT_INPUT_BYTES = 4 * 1024 * 1024;
export const MAX_DOCUMENT_NESTING_DEPTH = 32;
export const MAX_DOCUMENT_SCALARS = 100_000;

/** @param {unknown} value */
export function normalizeDocument(value) {
  inspectStructure(value, false);
  if (!isObject(value)) throw new Error("data document must be a JSON object");
  if (value.format === FORMAT) return parseCanonical(value);
  const variables = {};
  let scalarCount = 0;
  for (const [name, entry] of Object.entries(value)) {
    const variable = normalizePlainVariable(
      entry,
      name,
      MAX_DOCUMENT_SCALARS - scalarCount,
    );
    scalarCount += variable.values.length;
    requireScalarCount(scalarCount);
    defineVariable(variables, name, variable);
  }
  return { format: FORMAT, variables };
}

/** @param {string} text */
export function parseDocument(text) {
  return normalizeDocument(parseBoundedJson(text, false));
}

/** Parse one bounded JSON value used to construct a data document. */
export function parseDocumentValue(text) {
  return parseBoundedJson(text, true);
}

function parseBoundedJson(text, enforceScalarCount) {
  if (typeof text !== "string") throw new TypeError("data document JSON must be text");
  if (exceedsUtf8Bytes(text, MAX_DOCUMENT_INPUT_BYTES)) {
    throw new Error(
      `data document exceeds maximum UTF-8 input size of ${MAX_DOCUMENT_INPUT_BYTES} bytes`,
    );
  }
  let value;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  inspectStructure(value, enforceScalarCount);
  return value;
}

/** @param {ReturnType<typeof normalizeDocument>} document */
export function serializeDocument(document) {
  return `${JSON.stringify(document)}\n`;
}

function inspectStructure(value, enforceScalarCount) {
  let scalarCount = 0;
  let current = value;
  let depth = 0;
  const stack = [];
  while (true) {
    if (current !== null && typeof current === "object") {
      if (depth > MAX_DOCUMENT_NESTING_DEPTH) {
        throw new Error(
          `data document exceeds maximum nesting depth of ${MAX_DOCUMENT_NESTING_DEPTH}`,
        );
      }
      const children = Array.isArray(current) ? current : Object.values(current);
      if (children.length > 0) {
        stack.push({ children, index: 1, depth });
        current = children[0];
        depth += 1;
        continue;
      }
    } else {
      scalarCount += 1;
      if (enforceScalarCount) requireScalarCount(scalarCount);
    }
    let frame = stack.at(-1);
    while (frame !== undefined && frame.index >= frame.children.length) {
      stack.pop();
      frame = stack.at(-1);
    }
    if (frame === undefined) return;
    current = frame.children[frame.index++];
    depth = frame.depth + 1;
  }
}

function defineVariable(variables, name, value) {
  Object.defineProperty(variables, name, {
    value,
    enumerable: true,
    writable: true,
    configurable: true,
  });
}

function requireScalarCount(count) {
  if (count > MAX_DOCUMENT_SCALARS) {
    throw new Error(
      `data document exceeds maximum scalar count of ${MAX_DOCUMENT_SCALARS}`,
    );
  }
}

function parseCanonical(value) {
  const keys = Object.keys(value);
  if (keys.length !== 2 || !keys.includes("format") || !keys.includes("variables")) {
    throw new Error("canonical data document requires only format and variables");
  }
  if (!isObject(value.variables)) throw new Error("canonical variables must be an object");
  const variables = {};
  let scalarCount = 0;
  for (const [name, entry] of Object.entries(value.variables)) {
    const variable = parseCanonicalVariable(
      entry,
      name,
      MAX_DOCUMENT_SCALARS - scalarCount,
    );
    scalarCount += variable.values.length;
    requireScalarCount(scalarCount);
    defineVariable(variables, name, variable);
  }
  return { format: FORMAT, variables };
}

function parseCanonicalVariable(value, name, remainingScalars) {
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
  if (value.values.length > remainingScalars) {
    requireScalarCount(MAX_DOCUMENT_SCALARS + 1);
  }
  const expected = value.shape.length === 0 ? 1 : value.shape.reduce((a, b) => a * b, 1);
  if (value.values.length !== expected) {
    throw new Error(`variable ${name} shape expects ${expected} values, got ${value.values.length}`);
  }
  const values = value.values.map((entry) => canonicalScalar(entry, value.dtype, name));
  return { dtype: value.dtype, shape: [...value.shape], values };
}

function normalizePlainVariable(value, name, remainingScalars) {
  const { shape, values } = flatten(value, name, remainingScalars);
  const dtype = inferDtype(values, name);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = canonicalScalar(values[index], dtype, name);
  }
  return { dtype, shape, values };
}

// Determine the rectangular shape from the first branch, then visit each
// scalar exactly once. The traversal retains only one frame per nesting level.
function flatten(value, name, maximumScalars) {
  if (!Array.isArray(value)) {
    if (!validPlainScalar(value)) throw new Error(`variable ${name} must contain booleans or numbers`);
    if (maximumScalars < 1) requireScalarCount(MAX_DOCUMENT_SCALARS + 1);
    return { shape: [], values: [value] };
  }

  const shape = [];
  let first = value;
  while (Array.isArray(first)) {
    shape.push(first.length);
    if (first.length === 0) break;
    first = first[0];
  }
  if (!Array.isArray(first) && !validPlainScalar(first)) {
    throw new Error(`variable ${name} must contain booleans or numbers`);
  }

  const values = [];
  const arrays = [value];
  const indexes = [0];
  while (arrays.length > 0) {
    const depth = arrays.length - 1;
    const array = arrays[depth];
    if (array.length !== shape[depth]) {
      throw new Error(`variable ${name} arrays must be rectangular`);
    }
    if (indexes[depth] >= array.length) {
      arrays.pop();
      indexes.pop();
      continue;
    }
    const entry = array[indexes[depth]++];
    const childDepth = depth + 1;
    if (Array.isArray(entry)) {
      if (childDepth >= shape.length) {
        throw new Error(`variable ${name} arrays must be rectangular`);
      }
      arrays.push(entry);
      indexes.push(0);
      continue;
    }
    if (childDepth !== shape.length) {
      throw new Error(`variable ${name} arrays must be rectangular`);
    }
    if (!validPlainScalar(entry)) {
      throw new Error(`variable ${name} must contain booleans or numbers`);
    }
    values.push(entry);
    if (values.length > maximumScalars) {
      requireScalarCount(MAX_DOCUMENT_SCALARS + 1);
    }
  }
  return { shape, values };
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

function exceedsUtf8Bytes(value, maximumBytes) {
  let byteLength = 0;
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    byteLength += codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 :
      codePoint <= 0xffff ? 3 : 4;
    if (byteLength > maximumBytes) return true;
  }
  return false;
}

function isObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
