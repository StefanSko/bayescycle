import { renderRecoverySummary } from "/site/src/app/recovery.mjs";

function assert(condition, message) { if (!condition) throw new Error(message); }

export default [{
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
}];
