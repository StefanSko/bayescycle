import {
  EMPTY_RECOVERY_PROJECTION_NOTICE,
  SCENARIO_TARGET_DIVERGENCE_ERROR,
  assertScenarioCompatible,
  projectRecoveryTruth,
} from "/site/src/app/scenario.mjs";

const UTF8 = new TextEncoder();
const TEXT = new TextDecoder();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const SCHEMA = {
  schema_format: "bayescycle.playground.model-schema.v0",
  parameters: [
    { name: "alpha", prior: "Normal(0, 1)", constraint: null, shape: [], default: 0 },
    { name: "beta", prior: "Normal(0, 1)", constraint: null, shape: [], default: 0 },
  ],
  data: [{ name: "x", dtype: "float64", kind: "vector", length: null }],
  observed: [{ name: "y" }],
};

function canonicalVariables(variables) {
  return UTF8.encode(`${JSON.stringify({
    format: "bayescycle.data.json.v1",
    variables,
  })}\n`);
}

export default [
  {
    name: "scenario target divergence requires an explicit main-model recompile",
    fn: () => {
      let message = "";
      try {
        assertScenarioCompatible(
          { irHash: "a".repeat(64), modelSchema: SCHEMA },
          { targetIrHash: "b".repeat(64), modelSchema: SCHEMA },
        );
      } catch (error) { message = error.message; }
      assert(message === SCENARIO_TARGET_DIVERGENCE_ERROR,
        `target divergence error changed: ${message}`);
    },
  },
  {
    name: "scenario schema compatibility rejects added prior data slots",
    fn: () => {
      let message = "";
      try {
        assertScenarioCompatible(
          { irHash: "a".repeat(64), modelSchema: SCHEMA },
          {
            targetIrHash: "a".repeat(64),
            modelSchema: {
              ...SCHEMA,
              data: [
                { name: "prior_location", dtype: "float64", kind: "scalar", length: null },
                ...SCHEMA.data,
              ],
            },
          },
        );
      } catch (error) { message = error.message; }
      assert(message === "prior-only model must not declare data slots: prior_location",
        `scenario schema error changed: ${message}`);
    },
  },
  {
    name: "recovery truth projection retains shared parameters without mutating the pair",
    fn: () => {
      const original = canonicalVariables({
        location: { dtype: "float64", shape: [], values: [2.5] },
        beta: { dtype: "float64", shape: [], values: [3.1] },
        alpha: { dtype: "float64", shape: [], values: [0.2] },
      });
      const before = TEXT.decode(original);
      const projected = projectRecoveryTruth(original, SCHEMA);
      assert(projected.notice === null && projected.bytes instanceof Uint8Array,
        "shared recovery truth was skipped");
      const document = JSON.parse(TEXT.decode(projected.bytes));
      assert(Object.keys(document.variables).join(",") === "alpha,beta",
        `projected recovery names changed: ${Object.keys(document.variables)}`);
      assert(document.variables.beta.values[0] === 3.1, "shared beta truth changed");
      assert(TEXT.decode(original) === before, "paired parameter bytes were mutated");

      const empty = projectRecoveryTruth(
        canonicalVariables({
          location: { dtype: "float64", shape: [], values: [2.5] },
        }),
        SCHEMA,
      );
      assert(empty.bytes === null, "empty recovery projection produced bytes");
      assert(empty.notice === EMPTY_RECOVERY_PROJECTION_NOTICE,
        `empty recovery notice changed: ${empty.notice}`);
    },
  },
];
