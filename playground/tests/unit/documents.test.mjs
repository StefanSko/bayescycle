import {
  MAX_DOCUMENT_INPUT_BYTES,
  MAX_DOCUMENT_NESTING_DEPTH,
  MAX_DOCUMENT_SCALARS,
  normalizeDocument,
  parseDocument,
  serializeDocument,
} from "/site/src/data/documents.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function equal(actual, expected, message) {
  assert(JSON.stringify(actual) === JSON.stringify(expected), `${message}: ${JSON.stringify(actual)}`);
}

export default [
  {
    name: "normalizes plain scalar vector and matrix JSON",
    fn: () => {
      const document = normalizeDocument({ flag: true, n: 2, x: [1, 2.5], m: [[1, 2], [3, 4]] });
      equal(document, {
        format: "bayescycle.data.json.v1",
        variables: {
          flag: { dtype: "bool", shape: [], values: [true] },
          n: { dtype: "int64", shape: [], values: [2] },
          x: { dtype: "float64", shape: [2], values: [1, 2.5] },
          m: { dtype: "int64", shape: [2, 2], values: [1, 2, 3, 4] },
        },
      }, "plain normalization differs");
    },
  },
  {
    name: "preserves plain variables named format and variables",
    fn: () => {
      const document = normalizeDocument({ format: 7, variables: [1, 2] });
      equal(document.variables, {
        format: { dtype: "int64", shape: [], values: [7] },
        variables: { dtype: "int64", shape: [2], values: [1, 2] },
      }, "plain format/variables names changed");
    },
  },
  {
    name: "accepts and preserves every canonical dtype",
    fn: () => {
      const input = {
        format: "bayescycle.data.json.v1",
        variables: {
          b: { dtype: "bool", shape: [2], values: [true, false] },
          i32: { dtype: "int32", shape: [], values: [2147483647] },
          i64: { dtype: "int64", shape: [], values: [9007199254740991] },
          f32: { dtype: "float32", shape: [1], values: [0.5] },
          f64: { dtype: "float64", shape: [1], values: [1] },
        },
      };
      equal(normalizeDocument(input), input, "canonical document changed");
    },
  },
  {
    name: "rejects ragged arrays and invalid canonical values",
    fn: () => {
      for (const [input, expected] of [
        [{ x: [[1], [2, 3]] }, "rectangular"],
        [{ format: "bayescycle.data.json.v1", variables: { x: { dtype: "int64", shape: [2], values: [1] } } }, "shape"],
        [{ format: "bayescycle.data.json.v1", variables: { x: { dtype: "bool", shape: [], values: [1] } } }, "boolean"],
      ]) {
        let message = "";
        try { normalizeDocument(input); } catch (error) { message = String(error); }
        assert(message.includes(expected), `expected ${expected} error, got: ${message}`);
      }
    },
  },
  {
    name: "serializes canonical bytes deterministically",
    fn: () => {
      const text = serializeDocument(normalizeDocument({ x: [1, 2], y: 3.5 }));
      assert(text === '{"format":"bayescycle.data.json.v1","variables":{"x":{"dtype":"int64","shape":[2],"values":[1,2]},"y":{"dtype":"float64","shape":[],"values":[3.5]}}}\n', `unexpected bytes: ${text}`);
    },
  },
  {
    name: "rejects document byte depth and scalar caps with named errors",
    fn: () => {
      const exactBytes = `{}` + " ".repeat(MAX_DOCUMENT_INPUT_BYTES - 2);
      assert(Object.keys(parseDocument(exactBytes).variables).length === 0, "exact byte cap was rejected");
      let atDepth = 1;
      for (let depth = 0; depth < MAX_DOCUMENT_NESTING_DEPTH; depth += 1) atDepth = [atDepth];
      assert(normalizeDocument({ x: atDepth }).variables.x.shape.length === MAX_DOCUMENT_NESTING_DEPTH, "exact depth cap was rejected");

      let byteMessage = "";
      try { parseDocument(`{"x":"${"é".repeat(MAX_DOCUMENT_INPUT_BYTES)}"}`); } catch (error) { byteMessage = String(error); }
      assert(byteMessage.includes("UTF-8 input size") && byteMessage.includes(String(MAX_DOCUMENT_INPUT_BYTES)), `byte cap error was unclear: ${byteMessage}`);

      let nested = 1;
      for (let depth = 0; depth <= MAX_DOCUMENT_NESTING_DEPTH; depth += 1) nested = [nested];
      let depthMessage = "";
      try { normalizeDocument({ x: nested }); } catch (error) { depthMessage = String(error); }
      assert(depthMessage.includes("nesting depth") && depthMessage.includes(String(MAX_DOCUMENT_NESTING_DEPTH)), `depth cap error was unclear: ${depthMessage}`);

      let scalarMessage = "";
      try { normalizeDocument({ x: Array(MAX_DOCUMENT_SCALARS + 1).fill(1) }); } catch (error) { scalarMessage = String(error); }
      assert(scalarMessage.includes("scalar count") && scalarMessage.includes(String(MAX_DOCUMENT_SCALARS)), `scalar cap error was unclear: ${scalarMessage}`);
    },
  },
  {
    name: "normalizes a near-cap document in one flat value pass",
    fn: () => {
      const document = normalizeDocument({ x: Array(MAX_DOCUMENT_SCALARS).fill(7) });
      assert(document.variables.x.values.length === MAX_DOCUMENT_SCALARS, "near-cap values were truncated");
      assert(document.variables.x.shape[0] === MAX_DOCUMENT_SCALARS, "near-cap shape changed");
      assert(document.variables.x.values.at(-1) === 7, "near-cap tail changed");
    },
  },
  {
    name: "canonical document round-trips at the scalar limit",
    fn: () => {
      const normalized = normalizeDocument({ x: Array(MAX_DOCUMENT_SCALARS).fill(7) });
      const reparsed = parseDocument(serializeDocument(normalized));
      assert(reparsed.variables.x.values.length === MAX_DOCUMENT_SCALARS, "canonical scalar boundary was not closed");
      assert(serializeDocument(reparsed) === serializeDocument(normalized), "canonical boundary bytes changed");
    },
  },
  {
    name: "normalization preserves proto-named variables byte-exactly",
    fn: () => {
      const expected = '{"format":"bayescycle.data.json.v1","variables":{"__proto__":{"dtype":"int64","shape":[],"values":[1]}}}\n';
      const plain = serializeDocument(parseDocument('{"__proto__":1}'));
      assert(plain === expected, `plain __proto__ variable changed: ${plain}`);
      const canonical = serializeDocument(parseDocument(expected));
      assert(canonical === expected, `canonical __proto__ variable changed: ${canonical}`);
    },
  },
  {
    name: "compact eight schools sidecar preserves previous normalized values",
    fn: async () => {
      const [compact, previousEnvelope] = await Promise.all([
        fetch("/site/examples/eight_schools_non_centered.data.json").then(
          (response) => response.text(),
        ),
        fetch("/tests/fixtures/engine/eight_schools_non_centered/data.json").then(
          (response) => response.text(),
        ),
      ]);
      const actual = parseDocument(compact);
      const expected = parseDocument(previousEnvelope);
      assert(
        JSON.stringify(Object.keys(actual.variables)) ===
          JSON.stringify(Object.keys(expected.variables)),
        "eight schools variable order changed",
      );
      for (const name of Object.keys(expected.variables)) {
        assert(
          JSON.stringify(actual.variables[name].shape) ===
            JSON.stringify(expected.variables[name].shape) &&
            JSON.stringify(actual.variables[name].values) ===
              JSON.stringify(expected.variables[name].values),
          `eight schools ${name} shape or values changed`,
        );
      }
    },
  },
  {
    name: "representative normalization matches pinned canonical bytes",
    fn: async () => {
      const [input, expected] = await Promise.all([
        fetch("/tests/fixtures/document-normalization.input.json").then((response) => response.text()),
        fetch("/tests/fixtures/document-normalization.canonical.json").then((response) => response.text()),
      ]);
      const actual = serializeDocument(parseDocument(input));
      assert(actual === expected, `pinned canonical bytes changed: ${actual}`);
    },
  },
];
