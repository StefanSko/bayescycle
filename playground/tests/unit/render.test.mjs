import {
  average,
  divergenceVerdict,
  MAX_DASHBOARD_PARAMETER_COMPONENTS,
  prepareDashboardPlots,
  gaussianKde,
  quantile,
  rankHistogram,
  readDashboardData,
  renderDivergences,
  renderEnergy,
  renderEssRhat,
  renderPrecis,
  renderTrank,
} from "/site/src/dashboard/index.mjs";
import {
  overlapCoefficient,
  renderDensityOverlay,
  renderPriorPosteriorOverlay,
} from "/site/src/critique/render.mjs";

const FIXTURE_ROOT = "/tests/fixtures/engine/eight_schools_non_centered/";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function fetchText(path) {
  const response = await fetch(path);
  assert(response.ok, `fixture request for ${path} failed with HTTP ${response.status}`);
  return response.text();
}

let fixturePromise;
function fixture() {
  fixturePromise ??= Promise.all([
    fetchText(`${FIXTURE_ROOT}posterior.ndjson`),
    fetchText(`${FIXTURE_ROOT}diagnostics.json`),
  ]).then(([ndjson, diagnostics]) => ({
    data: readDashboardData({ fits: [ndjson], diagnose: diagnostics }),
    ndjson,
  }));
  return fixturePromise;
}

function parsedSvg(source) {
  assert(source.startsWith("<svg xmlns="), "renderer did not return an SVG root");
  assert(source.includes('role="img"'), "SVG has no image role");
  const document = new DOMParser().parseFromString(source, "image/svg+xml");
  assert(document.querySelectorAll("parsererror").length === 0, "SVG did not parse");
  const root = document.documentElement;
  assert(root.getAttribute("aria-label")?.trim(), "SVG has no non-empty accessible label");
  return document;
}

function occurrences(source, needle) {
  return source.split(needle).length - 1;
}

function normalDraws(count, location = 0, scale = 1) {
  return Array.from({ length: count }, (_, index) => {
    const u1 = (Math.floor(index / 20) + 0.5) / Math.ceil(count / 20);
    const u2 = ((index % 20) + 0.5) / 20;
    return location + scale * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  });
}

export default [
  {
    name: "readDashboardData shapes the fixture",
    fn: async () => {
      const { data, ndjson } = await fixture();
      const draws = ndjson.split("\n").filter(Boolean).slice(1, -1).map(JSON.parse);
      const chainIds = [...new Set(draws.map((draw) => draw.chain))];
      const drawsPerChain = chainIds.map(
        (chain) => draws.filter((draw) => draw.chain === chain).length,
      );
      assert(data.parameters.length > 0, "fixture has no parameters");
      for (const parameter of data.parameters) {
        assert(parameter.label.length > 0, `parameter ${parameter.name} has no label`);
        assert(
          parameter.chains.length === 2 &&
            parameter.chains.every((chain) => chain.length === 300),
          `${parameter.label} does not contain the fixture's two 300-draw chains`,
        );
      }
      assert(data.drawCount === 600, `expected 600 draws, got ${data.drawCount}`);
      assert(data.chainCount === 2, `expected 2 chains, got ${data.chainCount}`);
      assert(
        JSON.stringify(data.energies.map((values) => values.length)) ===
          JSON.stringify(drawsPerChain),
        "energy chain lengths differ from the raw fixture",
      );
      const divergenceCount = draws.filter((draw) => draw.diverging === true).length;
      assert(Number.isInteger(data.divergences), "divergence count is not an integer");
      assert(data.divergences >= 0, "divergence count is negative");
      assert(
        data.divergences === divergenceCount,
        `expected ${divergenceCount} divergences, got ${data.divergences}`,
      );
    },
  },
  {
    name: "dashboard plot preparation omits SVGs above component cap",
    fn: () => {
      assert(
        MAX_DASHBOARD_PARAMETER_COMPONENTS === 200,
        `dashboard component cap changed: ${MAX_DASHBOARD_PARAMETER_COMPONENTS}`,
      );
      const componentCount = MAX_DASHBOARD_PARAMETER_COMPONENTS + 1;
      const coordinates = Array.from({ length: componentCount }, (_, index) => [index]);
      const values = Array.from({ length: componentCount }, (_, index) => index / 10);
      const posterior = [
        JSON.stringify({ params: [{ name: "z", coordinate_order: coordinates }] }),
        JSON.stringify({ chain: 0, values: { z: values }, energy: 0, diverging: false }),
        JSON.stringify({ trailer: {} }),
      ].join("\n");
      const data = readDashboardData({ fits: [posterior], diagnose: {} });
      assert(data.parameters.length === componentCount, "expanded component count differs");
      const prepared = prepareDashboardPlots(data);
      assert(prepared.kind === "omitted", "oversized dashboard was rendered");
      assert(prepared.componentCount === componentCount, "omission count differs");
      assert(
        prepared.notice ===
          "Posterior has 201 parameter components; plots are shown for models with at most 200.",
        `dashboard omission notice differs: ${prepared.notice}`,
      );
      assert(!Object.hasOwn(prepared, "trank"), "oversized trank SVG was built");
      assert(!Object.hasOwn(prepared, "precis"), "oversized precis SVG was built");
    },
  },
  {
    name: "stats primitives",
    fn: () => {
      const integers = Array.from({ length: 100 }, (_, index) => index + 1);
      assert(quantile(integers, 0.5) === 50.5, "median of 1..100 is not 50.5");
      assert(average(integers) === 50.5, "average of 1..100 is not 50.5");
      const chains = [integers.slice(0, 40), integers.slice(40)];
      const histograms = rankHistogram(chains);
      assert(histograms.length === 2, "rank histogram did not return one row per chain");
      assert(histograms.every((bins) => bins.length === 20), "rank histogram needs 20 bins");
      assert(
        histograms.every(
          (bins, index) =>
            bins.reduce((sum, count) => sum + count, 0) === chains[index].length,
        ),
        "rank-bin counts do not sum to chain lengths",
      );
      const density = gaussianKde(normalDraws(100));
      assert(density.length >= 32, `KDE grid is too short: ${density.length}`);
      assert(
        density.every((point) => Number.isFinite(point.x) && point.density >= 0),
        "KDE contains an invalid point",
      );
    },
  },
  {
    name: "every renderer is deterministic",
    fn: async () => {
      const { data } = await fixture();
      const renderers = [
        () => renderTrank(data),
        () => renderTrank(data, "trace"),
        () => renderEssRhat(data),
        () => renderPrecis(data),
        () => renderEnergy(data),
        () => renderDivergences(data),
      ];
      for (const render of renderers) {
        const first = render();
        assert(first === render(), "renderer output is not byte-identical across calls");
        parsedSvg(first);
      }
    },
  },
  {
    name: "trank grid draws one panel per parameter",
    fn: async () => {
      const { data } = await fixture();
      assert(
        occurrences(renderTrank(data), "n_eff") === data.parameters.length,
        "trank panel count differs from the parameter count",
      );
    },
  },
  {
    name: "ess×r-hat scatter has the verdict line",
    fn: async () => {
      const { data } = await fixture();
      const source = renderEssRhat(data);
      const available = data.reportParameters.filter(
        (parameter) => parameter.ess !== null && parameter.rhat !== null,
      );
      assert(source.includes("0.1 N = 60"), "ESS verdict guide is absent");
      assert(
        parsedSvg(source).querySelectorAll("circle").length === available.length,
        "ESS scatter does not have one point per available parameter",
      );
    },
  },
  {
    name: "precis shows mean and 89% interval",
    fn: async () => {
      const { data } = await fixture();
      const source = renderPrecis(data);
      assert(source.includes("89% interval"), "precis interval heading is absent");
      const text = [...parsedSvg(source).querySelectorAll("text")].map((node) => node.textContent);
      assert(
        data.parameters.every(
          (parameter) => text.filter((value) => value === parameter.label).length === 1,
        ),
        "precis does not have exactly one labeled row per parameter",
      );
    },
  },
  {
    name: "precis labels its value axis and recovery truth",
    fn: async () => {
      const { data } = await fixture();
      const label = data.parameters[0].label;
      const source = renderPrecis(data, { [label]: 0 });
      const document = parsedSvg(source);
      assert(document.querySelector(".value-axis") !== null, "precis value baseline is absent");
      assert(
        document.querySelectorAll(".value-axis-tick").length >= 2 &&
          document.querySelectorAll(".value-axis-label").length >= 2,
        "precis min/zero/max ticks are not labelled",
      );
      assert(
        source.includes("posterior 89% interval · mean") && source.includes("true value"),
        "precis legend does not distinguish posterior and truth",
      );
      assert(
        document.querySelector(`.truth-marker[data-parameter="${label}"]`) !== null,
        "precis truth marker is absent",
      );
      assert(source.includes("· true 0.00"), "precis true-value annotation is absent");

      const observedSource = renderPrecis(data);
      assert(!observedSource.includes("truth-marker"), "observed precis acquired a truth marker");
      assert(!observedSource.includes("true value"), "observed precis acquired a truth legend");
    },
  },
  {
    name: "divergence verdict thresholds",
    fn: () => {
      assert(divergenceVerdict(0) === "outlook good", "zero-divergence verdict differs");
      assert(divergenceVerdict(5) === "roll again", "middle divergence verdict differs");
      assert(
        divergenceVerdict(10) === "check yourself before you wreck yourself",
        "high divergence verdict differs",
      );
    },
  },
  {
    name: "density overlay renders observed vs replicates",
    fn: () => {
      const observed = normalDraws(50);
      const input = {
        observed,
        replicates: Array.from({ length: 20 }, (_, replicate) =>
          observed.map((value, row) => value + Math.sin((replicate + 1) * (row + 1)) / 10),
        ),
        labels: observed.map((_, index) => `row ${index + 1}`),
      };
      const source = renderDensityOverlay(input);
      assert(source === renderDensityOverlay(input), "density overlay is not deterministic");
      const document = parsedSvg(source);
      assert(document.querySelectorAll("path").length === 21, "expected 20 replicate paths and 1 observed path");
      assert(document.querySelectorAll("line").length === 1, "density overlay axis is absent");
    },
  },
  {
    name: "prior→posterior overlay labels the overlap",
    fn: () => {
      const posterior = normalDraws(200);
      const matched = { kind: "normal", location: 0, scale: 1 };
      const wide = { kind: "normal", location: 0, scale: 100 };
      const matchedSource = renderPriorPosteriorOverlay([
        { name: "theta", prior: matched, posterior },
      ]);
      assert(matchedSource === renderPriorPosteriorOverlay([
        { name: "theta", prior: matched, posterior },
      ]), "prior-to-posterior overlay is not deterministic");
      parsedSvg(matchedSource);
      assert(/overlap \d+%/u.test(matchedSource), "matched overlay has no overlap percentage");
      assert(matchedSource.includes("data uninformative"), "matched overlay lacks its note");

      const wideSource = renderPriorPosteriorOverlay([
        { name: "theta", prior: wide, posterior },
      ]);
      const percentage = /overlap (\d+)%/u.exec(wideSource);
      assert(percentage !== null, "wide overlay has no overlap percentage");
      assert(Number(percentage[1]) < 20, `wide-prior overlap is not below 20%: ${percentage[1]}`);
      assert(!wideSource.includes("data uninformative"), "wide prior was called uninformative");

      const matchedCoefficient = overlapCoefficient(matched, posterior);
      const wideCoefficient = overlapCoefficient(wide, posterior);
      assert(
        [matchedCoefficient, wideCoefficient].every(
          (value) => typeof value === "number" && value >= 0 && value <= 1,
        ),
        "overlap coefficient is outside [0, 1]",
      );
      assert(matchedCoefficient > wideCoefficient, "matched overlap is not higher than wide overlap");
    },
  },
];
