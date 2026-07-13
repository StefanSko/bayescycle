import { exampleAssetUrl } from "/site/src/app/examples.mjs";
import { importJson, requiredInputs } from "/site/src/data/index.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function dataDocument(variables) {
  return JSON.stringify({ format: "bayescycle.data.json.v1", variables });
}

export default [
  {
    name: "importJson accepts all additional v1 dtypes",
    fn: () => {
      const document = importJson(dataDocument({
        flag: { dtype: "bool", shape: [2], values: [true, false] },
        count: { dtype: "int32", shape: [2], values: [1, 2] },
        measure: { dtype: "float32", shape: [2], values: [1.25, 2.5] },
      }));
      assert(document.variables.flag.dtype === "bool", "bool dtype was not retained");
      assert(document.variables.count.dtype === "int32", "int32 dtype was not retained");
      assert(document.variables.measure.dtype === "float32", "float32 dtype was not retained");

      let message = "";
      try {
        importJson(dataDocument({ mystery: { dtype: "uint8", shape: [1], values: [1] } }));
      } catch (error) {
        message = String(error instanceof Error ? error.message : error);
      }
      assert(message.includes("mystery") && message.includes("dtype"), `unclear dtype error: ${message}`);
    },
  },
  {
    name: "requiredInputs preserves every matrix dimension",
    fn: () => {
      const inputs = requiredInputs({
        model: {
          data: [{
            name: "x",
            value: { schema: { dims: [{ name: "n" }, { name: "m" }] } },
          }],
          observed_nodes: [],
        },
      });
      assert(inputs.length === 1, `expected one input, got ${inputs.length}`);
      assert(inputs[0].kind === "vector", `matrix kind differs: ${inputs[0].kind}`);
      assert(JSON.stringify(inputs[0].dims) === '["n","m"]', `matrix dims differ: ${JSON.stringify(inputs[0].dims)}`);
    },
  },
  {
    name: "example asset URLs retain a deployment path prefix",
    fn: () => {
      const moduleUrl = "https://example.test/project/site/src/app/main.mjs";
      const manifest = exampleAssetUrl("EXAMPLES.json", moduleUrl);
      const source = exampleAssetUrl("linear_regression.py", moduleUrl);
      assert(
        manifest.href === "https://example.test/project/site/examples/EXAMPLES.json",
        `manifest URL lost its prefix: ${manifest.href}`,
      );
      assert(
        source.href === "https://example.test/project/site/examples/linear_regression.py",
        `source URL lost its prefix: ${source.href}`,
      );
    },
  },
];
