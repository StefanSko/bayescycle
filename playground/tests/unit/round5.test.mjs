import { designDocument } from "/site/src/app/design.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export default [
  {
    name: "designDocument emits validated integer scalars",
    fn: () => {
      const document = designDocument({ n_groups: { value: 2 } });
      const variable = document.variables.n_groups;
      assert(variable.dtype === "int64", `scalar dtype differs: ${variable.dtype}`);
      assert(JSON.stringify(variable.shape) === "[]", "scalar shape differs");
      assert(JSON.stringify(variable.values) === "[2]", "scalar value differs");

      let message = "";
      try {
        designDocument({ n_groups: { value: 0 } });
      } catch (error) {
        message = String(error instanceof Error ? error.message : error);
      }
      assert(message.includes("positive integer"), `unclear scalar validation: ${message}`);
    },
  },
];
