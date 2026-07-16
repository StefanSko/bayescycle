/*
 * Design expressions use a permanently pinned 32-bit PRNG. The seed is first
 * offset by the golden-ratio word, then each draw applies the two Math.imul
 * xor-mix rounds below. Uniform draws divide the unsigned word by 2^32;
 * normal draws use Box-Muller without caching. Changing this algorithm would
 * change authored design documents, so its exact outputs are frozen in tests.
 *
 * Determinism boundary: uniform draws use only integer mixing and one exact
 * binary division, so they are bit-stable everywhere. Box-Muller normal draws
 * go through Math.log/sqrt/cos, whose last bits are implementation-defined;
 * the serialized design DOCUMENT written at authoring time is therefore the
 * authoritative artifact, and carried documents (share links, portable runs)
 * are never re-derived from their expressions.
 */

const CALL = /^([a-z]+)\((.*)\)$/s;
const SAFE_FUNCTIONS = new Set(["linspace", "repeat", "normal", "uniform"]);
// Previews evaluate on every keystroke; the cap keeps a stray count from
// allocating an unbounded array before document-size limits can apply.
export const MAX_ELEMENTS = 100000;

export function evaluateDesignExpression(text) {
  if (typeof text !== "string") throw new TypeError("design expression must be text");
  const source = text.trim();
  if (source === "") throw new Error("enter a design expression");
  if (source.startsWith("[")) return literalArray(source);

  const call = source.match(CALL);
  if (call === null) throw new Error("expected fn(…) or a literal [v, …]");
  const [, name, argumentSource] = call;
  if (!SAFE_FUNCTIONS.has(name)) throw new Error(`unknown function ${name}(…)`);
  const { positional, named } = parseArguments(argumentSource);

  if (name === "linspace") return linspace(positional, named);
  if (name === "repeat") return repeat(positional, named);
  return randomValues(name, positional, named);
}

export function evaluateDesignSlots(expressions) {
  if (expressions === null || typeof expressions !== "object" || Array.isArray(expressions)) {
    throw new TypeError("design expressions must be an object");
  }
  return Object.fromEntries(
    Object.entries(expressions).map(([name, expression]) => [
      name,
      evaluateDesignExpression(expression),
    ]),
  );
}

function seededRng(seed) {
  let state = (seed >>> 0) + 0x9e3779b9;
  return () => {
    state = Math.imul(state ^ (state >>> 16), 0x21f0aaad);
    state = Math.imul(state ^ (state >>> 15), 0x735a2d97);
    state ^= state >>> 15;
    return (state >>> 0) / 4294967296;
  };
}

function literalArray(source) {
  const value = parseJson(source, "literal array");
  if (!Array.isArray(value)) throw new Error("literal design value must be an array");
  requireBoundedCount(value.length, "literal array length");
  validateNumericArray(value, "literal array");
  return value;
}

function parseArguments(source) {
  const positional = [];
  const named = {};
  for (const raw of splitArguments(source)) {
    const part = raw.trim();
    if (part === "") throw new Error("empty argument");
    const match = part.match(/^([a-z]+)\s*=\s*(.+)$/s);
    if (match === null) {
      positional.push(parseJson(part, "argument"));
      continue;
    }
    const [, name, valueSource] = match;
    if (Object.hasOwn(named, name)) throw new Error(`duplicate named argument ${name}`);
    named[name] = parseJson(valueSource, `named argument ${name}`);
  }
  return { positional, named };
}

function splitArguments(source) {
  if (source.trim() === "") return [];
  const parts = [];
  let start = 0;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') inString = false;
      continue;
    }
    if (character === '"') inString = true;
    else if (character === "[") depth += 1;
    else if (character === "]") {
      depth -= 1;
      if (depth < 0) throw new Error("unbalanced ] in arguments");
    } else if (character === "," && depth === 0) {
      parts.push(source.slice(start, index));
      start = index + 1;
    }
  }
  if (inString || depth !== 0) throw new Error("unbalanced argument syntax");
  parts.push(source.slice(start));
  return parts;
}

function linspace(positional, named) {
  requireNoNamed(named, "linspace");
  if (positional.length !== 3) {
    throw new Error("linspace(start, stop, n) needs 3 args, n ≥ 2");
  }
  const [start, stop, count] = positional;
  requireFinite(start, "linspace start");
  requireFinite(stop, "linspace stop");
  if (!Number.isSafeInteger(count) || count < 2) {
    throw new Error("linspace(start, stop, n) needs 3 args, n ≥ 2");
  }
  requireBoundedCount(count, "linspace n");
  return requireFiniteResults(Array.from(
    { length: count },
    (_, index) => start + (index * (stop - start)) / (count - 1),
  ), "linspace");
}

function repeat(positional, named) {
  requireNoNamed(named, "repeat");
  if (positional.length !== 2 || !Array.isArray(positional[0]) ||
      !Number.isSafeInteger(positional[1]) || positional[1] < 1) {
    throw new Error("repeat([v, …], times) needs an array and times ≥ 1");
  }
  validateNumericArray(positional[0], "repeat levels");
  requireBoundedCount(positional[0].length * positional[1], "repeat output length");
  return positional[0].flatMap((value) => Array(positional[1]).fill(value));
}

function randomValues(name, positional, named) {
  const signature = name === "normal" ? "normal(loc, scale, n, seed=<int>)" :
    "uniform(low, high, n, seed=<int>)";
  if (positional.length !== 3 || Object.keys(named).some((key) => key !== "seed")) {
    throw new Error(`${signature} needs 3 positional args and optional seed`);
  }
  const [first, second, count] = positional;
  requireFinite(first, `${name} first argument`);
  requireFinite(second, `${name} second argument`);
  if (!Number.isSafeInteger(count) || count < 1) throw new Error(`${name} n must be an integer ≥ 1`);
  requireBoundedCount(count, `${name} n`);
  const seed = Object.hasOwn(named, "seed") ? named.seed : 0;
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error(`${name} seed must be an integer in 0..4294967295`);
  }
  if (name === "normal" && !(second > 0)) throw new Error("normal scale must be greater than 0");
  if (name === "uniform" && !(second > first)) throw new Error("uniform high must be greater than low");

  const random = seededRng(seed);
  if (name === "uniform") {
    return requireFiniteResults(
      Array.from({ length: count }, () => first + (second - first) * random()),
      name,
    );
  }
  return requireFiniteResults(Array.from({ length: count }, () => {
    const u = Math.max(random(), 1e-12);
    const v = random();
    return first + second * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }), name);
}

function parseJson(source, label) {
  try {
    return JSON.parse(source);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`${label} is not valid JSON: ${detail}`);
  }
}

function validateNumericArray(value, label) {
  for (const entry of value) {
    if (typeof entry !== "number" || !Number.isFinite(entry)) {
      throw new Error(`${label} must contain only finite numbers`);
    }
  }
}

function requireFinite(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
}

function requireFiniteResults(values, name) {
  if (!values.every(Number.isFinite)) throw new Error(`${name} produced non-finite values`);
  return values;
}

function requireNoNamed(named, name) {
  if (Object.keys(named).length !== 0) throw new Error(`${name} does not accept named arguments`);
}

function requireBoundedCount(count, label) {
  if (count > MAX_ELEMENTS) throw new Error(`${label} must be at most ${MAX_ELEMENTS}`);
}
