import { recoveryTruthMap, renderRecoverySummary } from "/site/src/app/recovery.mjs";

function assert(condition, message) { if (!condition) throw new Error(message); }

export default [
  {
    name: "recovery summary displays engine-owned facts",
    fn: () => {
      const html = renderRecoverySummary({
        interval: 0.8,
        target_order: ["alpha", "beta"],
        targets: {
          alpha: { truth: 0.2, mean: 0.3, lower: -0.1, upper: 0.7, interval_contains_truth: true },
          beta: { truth: 0.6, mean: 0.1, lower: -0.2, upper: 0.4, interval_contains_truth: false },
        },
      });
      assert(html.includes("alpha") && html.includes("beta"), "target labels missing");
      assert(html.includes("inside") && html.includes("outside"), "coverage verdicts missing");
      assert(html.includes("80% interval"), "interval probability missing");
    },
  },
  {
    name: "recovery truth map uses dashboard parameter labels",
    fn: () => {
      const truth = recoveryTruthMap({
        target_order: ["alpha", "beta"],
        targets: {
          alpha: { truth: 0.2 },
          beta: { truth: [0.5, 0.6, 0.7, 0.8] },
        },
      }, ["alpha", "beta[0,0]", "beta[0,1]", "beta[1,0]", "beta[1,1]"]);
      assert(truth.alpha === 0.2, "scalar truth label changed");
      assert(truth["beta[0,0]"] === 0.5, "first vectorized truth label changed");
      assert(truth["beta[1,1]"] === 0.8, "last vectorized truth label changed");
      assert(Object.keys(truth).length === 5, "truth map has unexpected labels");
    },
  },
  {
    name: "recovery truth indexing preserves vector labels at scale",
    fn: () => {
      const parameterCount = 250;
      const componentCount = 20;
      const targetOrder = Array.from(
        { length: parameterCount },
        (_, parameter) => `theta_${parameter}`,
      );
      const targets = Object.fromEntries(targetOrder.map((name, parameter) => [
        name,
        {
          truth: Array.from(
            { length: componentCount },
            (_, component) => parameter * componentCount + component,
          ),
        },
      ]));
      const labels = targetOrder.flatMap((name) =>
        Array.from({ length: componentCount }, (_, component) => `${name}[${component}]`));
      const truth = recoveryTruthMap({ target_order: targetOrder, targets }, labels);
      assert(
        Object.keys(truth).length === parameterCount * componentCount,
        "large truth map component count differs",
      );
      assert(truth["theta_0[0]"] === 0, "first large truth value changed");
      assert(truth["theta_127[9]"] === 2549, "middle large truth value changed");
      assert(truth["theta_249[19]"] === 4999, "last large truth value changed");
    },
  },
];
