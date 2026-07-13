import { designDefaults, designDocument, truthDocument } from "/site/src/app/design.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertThrows(operation, message) {
  let threw = false;
  try {
    operation();
  } catch {
    threw = true;
  }
  assert(threw, message);
}

export default [
  {
    name: "scalar designs distinguish counts from continuous values",
    fn: () => {
      const ir = {
        model: {
          data: [
            { name: "offset", value: { schema: { dims: [] } } },
            { name: "n_groups", value: { schema: { dims: [] } } },
            { name: "x", value: { schema: { rank: 1 } } },
          ],
          params: [
            { name: "z", value: { size: { node: "DataRef", name: "n_groups" } } },
          ],
        },
      };
      const defaults = designDefaults(ir);
      assert(defaults.offset.value === 1, "continuous scalar default differs");
      assert(defaults.offset.countLike === false, "continuous scalar was marked count-like");
      assert(defaults.n_groups.value === 2, "count scalar default differs");
      assert(defaults.n_groups.countLike === true, "size scalar was not marked count-like");

      const continuous = designDocument({ offset: { value: -0.5, countLike: false } });
      assert(
        JSON.stringify(continuous.variables.offset) ===
          JSON.stringify({ dtype: "float64", shape: [], values: [-0.5] }),
        `continuous scalar document differs: ${JSON.stringify(continuous.variables.offset)}`,
      );
      assertThrows(
        () => designDocument({ n_groups: { value: 0, countLike: true } }),
        "count scalar accepted zero",
      );
      assertThrows(
        () => designDocument({ n_groups: { value: 1.5, countLike: true } }),
        "count scalar accepted a non-integer",
      );
    },
  },
  {
    name: "ordered vector truth is increasing and centered",
    fn: () => {
      const values = truthDocument(
        { cutpoints: 2 },
        { cutpoints: { size: 4, ordered: true } },
      ).variables.cutpoints.values;
      assert(values.length === 4, `ordered truth length differs: ${values.length}`);
      assert(
        values.every((value, index) => index === 0 || value > values[index - 1]),
        `ordered truth is not strictly increasing: ${JSON.stringify(values)}`,
      );
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      assert(mean === 2, `ordered truth is not centered on 2: ${String(mean)}`);
    },
  },
];
