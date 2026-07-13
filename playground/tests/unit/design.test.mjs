import { compile } from "/site/src/compile/index.mjs";
import {
  defaultTruth,
  designDefaults,
  designDocument,
  truthDocument,
} from "/site/src/app/design.mjs";
import { renderPriorPredictiveDensity } from "/site/src/critique/render.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertJson(actual, expected, message) {
  assert(JSON.stringify(actual) === JSON.stringify(expected), `${message}: ${JSON.stringify(actual)}`);
}

function parsedSvg(source) {
  assert(source.startsWith("<svg xmlns="), "renderer did not return an SVG root");
  const document = new DOMParser().parseFromString(source, "image/svg+xml");
  assert(document.querySelectorAll("parsererror").length === 0, "SVG did not parse");
  const root = document.documentElement;
  assert(root.getAttribute("role") === "img", "SVG has no image role");
  assert(root.getAttribute("aria-label") === "Prior predictive density", "SVG label differs");
  return document;
}

export default [
  {
    name: "designDocument builds decorrelated permuted-linspace envelopes",
    fn: () => {
      const five = designDocument({ M: { low: -1, high: 1, n: 5 } });
      assert(five.format === "bayescycle.data.json.v1", "five-point envelope format differs");
      const fiveEntry = five.variables.M;
      assertJson(fiveEntry.shape, [5], "five-point shape differs");
      assert(fiveEntry.dtype === "float64", "five-point dtype differs");
      assertJson(
        [...fiveEntry.values].sort((a, b) => a - b),
        [-1, -0.5, 0, 0.5, 1],
        "five-point values are not a permutation of the linspace",
      );
      assertJson(
        designDocument({ M: { low: -1, high: 1, n: 5 } }),
        five,
        "designDocument is not deterministic",
      );
      const pair = designDocument({
        M: { low: -1, high: 1, n: 50 },
        A: { low: -1, high: 1, n: 50 },
      }).variables;
      assert(
        JSON.stringify(pair.M.values) !== JSON.stringify(pair.A.values),
        "same-range variables must not produce identical (collinear) designs",
      );
      const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
      const mM = mean(pair.M.values);
      const mA = mean(pair.A.values);
      let covariance = 0;
      let varM = 0;
      let varA = 0;
      for (let index = 0; index < 50; index += 1) {
        covariance += (pair.M.values[index] - mM) * (pair.A.values[index] - mA);
        varM += (pair.M.values[index] - mM) ** 2;
        varA += (pair.A.values[index] - mA) ** 2;
      }
      const correlation = covariance / Math.sqrt(varM * varA);
      assert(
        Math.abs(correlation) < 0.5,
        `same-range design variables stay too correlated: r=${correlation}`,
      );
      assertJson(
        designDocument({ M: { low: 2, high: 9, n: 1 } }),
        {
          format: "bayescycle.data.json.v1",
          variables: { M: { dtype: "float64", shape: [1], values: [2] } },
        },
        "one-point design document differs",
      );
      let message = "";
      try {
        designDocument({ M: { low: -1, high: 1, n: 2.5 } });
      } catch (error) {
        message = error.message;
      }
      assert(
        message === "Design values for M require finite low/high and a positive integer n",
        `non-integer validation message differs: ${message}`,
      );
    },
  },
  {
    name: "truthDocument builds scalar envelopes",
    fn: () => {
      assertJson(
        truthDocument({ alpha: 0.5 }),
        {
          format: "bayescycle.data.json.v1",
          variables: { alpha: { dtype: "float64", shape: [], values: [0.5] } },
        },
        "truth document differs",
      );
      let message = "";
      try {
        truthDocument({ alpha: Number.NaN });
      } catch (error) {
        message = error.message;
      }
      assert(message === "Truth value for alpha must be finite", `NaN validation message differs: ${message}`);
    },
  },
  {
    name: "defaultTruth is constraint-aware",
    fn: async () => {
      const response = await fetch("/tests/fixtures/divorce/generative.py");
      assert(response.ok, `fixture request failed with HTTP ${response.status}`);
      const result = await compile(await response.text());
      assert(result.ok, `compile failed: ${result.message ?? "unknown error"}`);
      const ir = JSON.parse(new TextDecoder().decode(result.irBytes));
      assertJson(
        defaultTruth(ir),
        { alpha: 0, beta_m: 0, beta_a: 0, sigma: 1 },
        "constraint-aware truth defaults differ",
      );
      // Positive default range: design values feed arbitrary data slots —
      // including scale parameters like eight-schools sigma — and negative
      // scales make the engine reject the out-of-the-box run.
      assertJson(
        designDefaults(ir),
        { M: { low: 0.5, high: 1.5, n: 50 }, A: { low: 0.5, high: 1.5, n: 50 } },
        "vector design defaults differ",
      );
    },
  },
  {
    name: "renderPriorPredictiveDensity is deterministic",
    fn: () => {
      const replicates = Array.from({ length: 20 }, (_, replicate) =>
        Array.from({ length: 50 }, (_, index) =>
          Math.sin((replicate + 1) * (index + 1) / 17) + replicate / 20,
        ),
      );
      const source = renderPriorPredictiveDensity(replicates);
      assert(source === renderPriorPredictiveDensity(replicates), "renderer output is not byte-identical");
      const document = parsedSvg(source);
      assert(document.querySelectorAll("path").length === 20, "expected one density path per replicate");
    },
  },
];
