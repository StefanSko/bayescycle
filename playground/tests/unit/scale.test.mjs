import { subsampleReplicates } from "/site/src/app/subsample.mjs";
import { renderDensityOverlay } from "/site/src/critique/render.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function predictiveInput(replicates) {
  const observed = Array.from({ length: 8 }, (_, index) => index / 3 - 1);
  return {
    observed,
    labels: observed.map((_, index) => `row ${index + 1}`),
    replicates,
  };
}

const replicates = Array.from({ length: 2000 }, (_, replicate) =>
  Array.from({ length: 8 }, (_, index) =>
    Math.sin((replicate + 1) * (index + 1) / 17) + replicate / 2000,
  ),
);

function pathCount(source) {
  const document = new DOMParser().parseFromString(source, "image/svg+xml");
  assert(document.querySelectorAll("parsererror").length === 0, "SVG did not parse");
  return document.querySelectorAll("path").length;
}

export default [
  {
    name: "app subsampling caps a 2000-replicate density overlay",
    fn: () => {
      const sampled = subsampleReplicates(replicates);
      assert(sampled.length === 50, `expected 50 sampled replicates, got ${sampled.length}`);
      const source = renderDensityOverlay(predictiveInput(sampled));
      assert(pathCount(source) <= 51, "subsampled overlay rendered more than 51 paths");
    },
  },
  {
    name: "density overlay accepts 2000 replicates directly",
    fn: () => {
      const source = renderDensityOverlay(predictiveInput(replicates));
      assert(pathCount(source) === 2001, "direct overlay did not render every replicate");
    },
  },
];
