import { truthDocument } from "/site/src/app/design.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export default [
  {
    name: "truthDocument broadcasts resolved vector parameters",
    fn: () => {
      const scalar = truthDocument({ tau: 1 }, {});
      assert(
        JSON.stringify(scalar.variables.tau) ===
          JSON.stringify({ dtype: "float64", shape: [], values: [1] }),
        `scalar truth changed: ${JSON.stringify(scalar.variables.tau)}`,
      );
      const vector = truthDocument({ z: 0.5 }, { z: 8 });
      assert(
        JSON.stringify(vector.variables.z) ===
          JSON.stringify({ dtype: "float64", shape: [8], values: Array(8).fill(0.5) }),
        `vector truth differs: ${JSON.stringify(vector.variables.z)}`,
      );
    },
  },
];
