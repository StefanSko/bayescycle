import { normalizeDocument, serializeDocument } from "/site/src/data/documents.mjs";

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
];
