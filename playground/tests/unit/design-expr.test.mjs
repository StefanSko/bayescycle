import {
  evaluateDesignExpression,
  evaluateDesignSlots,
} from "/site/src/generation/design-expr.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function equal(actual, expected, label) {
  assert(JSON.stringify(actual) === JSON.stringify(expected),
    `${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

function rejects(source, expected) {
  let detail = "";
  try { evaluateDesignExpression(source); }
  catch (error) { detail = String(error); }
  assert(detail.includes(expected), `${source} did not fail with ${expected}: ${detail}`);
}

export default [
  {
    name: "evaluates literal linspace and repeat expressions exactly",
    fn: () => {
      equal(evaluateDesignExpression("[1, 2.5, -3]"), [1, 2.5, -3], "literal");
      equal(evaluateDesignExpression("linspace(-2, 2, 5)"), [-2, -1, 0, 1, 2], "linspace");
      equal(evaluateDesignExpression("repeat([1, 2.5], 2)"), [1, 1, 2.5, 2.5], "repeat");
      equal(evaluateDesignSlots({ x: "linspace(0, 1, 3)", z: "[4]" }),
        { x: [0, 0.5, 1], z: [4] }, "slots");
    },
  },
  {
    name: "pins seeded uniform values",
    fn: () => {
      equal(evaluateDesignExpression("uniform(-1, 3, 4, seed=7)"), [
        2.5747826183214784,
        2.406814116984606,
        1.233479361049831,
        1.8590229842811823,
      ], "seeded uniform");
    },
  },
  {
    name: "pins seeded normal values and default seed",
    fn: () => {
      equal(evaluateDesignExpression("normal(0, 1, 4, seed=7)"), [
        0.2827641331720083,
        -0.23711713080514216,
        -0.10630266734442814,
        -0.061663084550491,
      ], "seeded normal");
      equal(evaluateDesignExpression("normal(2, 0.5, 3)"), [
        2.601063224324561,
        2.6785726213783603,
        1.9815311023134234,
      ], "default-seed normal");
    },
  },
  {
    name: "rejects malformed and out-of-vocabulary expressions",
    fn: () => {
      rejects("", "enter a design expression");
      rejects("range(1, 3)", "unknown function");
      rejects("linspace(0, 1, 1)", "n ≥ 2");
      rejects("repeat([1], 0)", "times ≥ 1");
      rejects("normal(0, 0, 2)", "scale must be greater");
      rejects("uniform(2, 1, 2)", "high must be greater");
      rejects("normal(0, 1, 2, seed=-1)", "seed must be an integer");
      rejects("[1, true]", "finite numbers");
      rejects("linspace(0,,2)", "empty argument");
      rejects("linspace(-1e308, 1e308, 3)", "produced non-finite values");
      rejects("uniform(-1e308, 1e308, 3, seed=1)", "produced non-finite values");
      rejects("normal(1e308, 1e308, 3, seed=1)", "produced non-finite values");
      rejects("linspace(0, 1, 100001)", "at most 100000");
      rejects("repeat([1, 2], 60000)", "at most 100000");
      rejects("normal(0, 1, 1000000000)", "at most 100000");
      rejects("uniform(0, 1, 1000000000)", "at most 100000");
      rejects(`[${"1, ".repeat(100000)}1]`, "at most 100000");
    },
  },
];
