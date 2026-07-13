// Source: src/critique/render.ts, bayesledger @ 7346d71.

import { gaussianKde } from "../dashboard/stats.mjs";

const FONT = "font-family:var(--mono);font-size:12px";
const INK = "var(--ink)";
const MUTED = "var(--muted)";
const LINE = "var(--line-strong)";
const ACCENT = "var(--accent)";

export function renderPriorPredictiveDensity(replicates) {
  if (replicates.length === 0 || replicates.some((values) => values.length === 0)) {
    throw new Error("Prior predictive draws are required");
  }
  const densities = replicates.map((values) => gaussianKde(values));
  const xs = densities.flatMap((density) => density.map((point) => point.x));
  const ys = densities.flatMap((density) => density.map((point) => point.density));
  const [minX, maxX] = extent(xs);
  const maxY = maximum(ys, 0.001);
  const px = scale(minX, maxX, 48, 520);
  const py = scale(0, maxY, 220, 34);
  const paths = densities
    .map(
      (density) =>
        `<path d="${path(density.map((point) => [px(point.x), py(point.density)]))}" fill="none" stroke="${ACCENT}" opacity="0.24"/>`,
    )
    .join("");
  return svg(
    540,
    260,
    "Prior predictive density",
    `<text x="12.00" y="20.00" fill="${INK}">prior predictive densities</text><line x1="48.00" y1="220.00" x2="520.00" y2="220.00" stroke="${LINE}"/>${paths}<text x="48.00" y="242.00" fill="${MUTED}">${minX.toPrecision(3)}</text><text x="478.00" y="242.00" fill="${MUTED}">${maxX.toPrecision(3)}</text>`,
  );
}

export function renderDensityOverlay(data) {
  validatePredictive(data);
  const observed = gaussianKde(data.observed);
  const replicated = data.replicates.map((values) => gaussianKde(values));
  const all = [observed, ...replicated];
  const xs = all.flatMap((density) => density.map((point) => point.x));
  const ys = all.flatMap((density) => density.map((point) => point.density));
  const [minX, maxX] = extent(xs);
  const maxY = maximum(ys, 0.001);
  const px = scale(minX, maxX, 48, 520);
  const py = scale(0, maxY, 220, 34);
  const replicatePaths = replicated
    .map(
      (density) =>
        `<path d="${path(density.map((point) => [px(point.x), py(point.density)]))}" fill="none" stroke="${MUTED}" opacity="0.28"/>`,
    )
    .join("");
  return svg(
    540,
    260,
    "Density overlay",
    `<text x="12.00" y="20.00" fill="${INK}">observed density · replicated densities</text><line x1="48.00" y1="220.00" x2="520.00" y2="220.00" stroke="${LINE}"/>${replicatePaths}<path d="${path(observed.map((point) => [px(point.x), py(point.density)]))}" fill="none" stroke="${ACCENT}" stroke-width="2.5"/><text x="48.00" y="242.00" fill="${MUTED}">${minX.toPrecision(3)}</text><text x="478.00" y="242.00" fill="${MUTED}">${maxX.toPrecision(3)}</text>`,
  );
}

export function renderPriorPosteriorOverlay(parameters) {
  if (parameters.length === 0) {
    throw new Error("At least one parameter overlay is required");
  }
  const panels = parameters
    .map((parameter, index) => {
      if (parameter.posterior.length === 0) {
        throw new Error(`Parameter '${parameter.name}' has no posterior draws`);
      }
      const densities = overlayDensities(parameter.prior, parameter.posterior);
      const { posterior, prior, minX, maxX } = densities;
      const maxY = maximum(
        prior.map((point) => point.density),
        maximum(posterior.map((point) => point.density), 0.001),
      );
      const top = 43 + index * 160;
      const bottom = top + 105;
      const px = scale(minX, maxX, 58, 518);
      const py = scale(0, maxY, bottom, top);
      const coefficient = densityOverlap(prior, posterior);
      const note = coefficient > 0.8 ? " · data uninformative" : "";
      return `<text x="12.00" y="${f(top - 12)}" fill="${INK}">${esc(parameter.name)} · overlap ${String(Math.round(coefficient * 100))}%${note}</text><path d="${path(prior.map((point) => [px(point.x), py(point.density)]))}" fill="none" stroke="${MUTED}" stroke-dasharray="5 4"/><path d="${path(posterior.map((point) => [px(point.x), py(point.density)]))}" fill="none" stroke="${ACCENT}" stroke-width="2.5"/>`;
    })
    .join("");
  return svg(540, 30 + parameters.length * 160, "Prior to posterior overlay", panels);
}

/** Return the normalized density-overlap coefficient for a prior and posterior draws. */
export function overlapCoefficient(prior, posterior) {
  const densities = overlayDensities(prior, posterior);
  return densityOverlap(densities.prior, densities.posterior);
}

function overlayDensities(priorValue, posteriorValues) {
  const posterior = gaussianKde(posteriorValues);
  const sampledValues = priorSamples(priorValue);
  const analyticPrior = sampledValues === undefined ? priorValue : undefined;
  const sampledPrior = sampledValues === undefined ? undefined : gaussianKde(sampledValues);
  const posteriorXs = posterior.map((point) => point.x);
  const priorXs =
    sampledPrior?.map((point) => point.x) ??
    analyticGrid(analyticPrior ?? { kind: "normal" }, posteriorXs);
  const [minX, maxX] = extent(posteriorXs.concat(priorXs));
  const xs = grid(minX, maxX);
  const prior =
    sampledPrior ??
    xs.map((x) => ({
      x,
      density: priorDensity(analyticPrior ?? { kind: "normal" }, x),
    }));
  return { posterior, prior, minX, maxX };
}

function validatePredictive(data) {
  if (
    data.observed.length === 0 ||
    data.labels.length !== data.observed.length ||
    data.replicates.length === 0 ||
    data.replicates.some((row) => row.length !== data.observed.length)
  ) {
    throw new Error("Predictive check data dimensions do not agree");
  }
}

function priorSamples(value) {
  return Array.isArray(value) ? value : undefined;
}

function priorDensity(prior, x) {
  const scaleValue = prior.scale ?? 1;
  if (prior.kind === "normal") {
    return (
      Math.exp(-0.5 * ((x - (prior.location ?? 0)) / scaleValue) ** 2) /
      (scaleValue * Math.sqrt(2 * Math.PI))
    );
  }
  if (prior.kind === "half-normal") {
    return x < 0
      ? 0
      : (Math.sqrt(2 / Math.PI) / scaleValue) *
          Math.exp(-0.5 * (x / scaleValue) ** 2);
  }
  const rate = prior.rate ?? 1;
  return x < 0 ? 0 : rate * Math.exp(-rate * x);
}

function analyticGrid(prior, posterior) {
  const scaleValue = prior.scale ?? 1 / (prior.rate ?? 1);
  const location = prior.kind === "normal" ? (prior.location ?? 0) : 0;
  return grid(
    minimum(posterior, location - (prior.kind === "normal" ? 4 : 0) * scaleValue),
    maximum(posterior, location + 6 * scaleValue),
  );
}

function densityOverlap(prior, posterior) {
  const min = Math.min(prior[0]?.x ?? 0, posterior[0]?.x ?? 0);
  const max = Math.max(prior.at(-1)?.x ?? 1, posterior.at(-1)?.x ?? 1);
  const xs = grid(min, max);
  const priorValues = interpolate(prior, xs);
  const posteriorValues = interpolate(posterior, xs);
  const priorArea = integral(xs, priorValues);
  const posteriorArea = integral(xs, posteriorValues);
  if (priorArea <= 0 || posteriorArea <= 0) return 0;
  const coefficient = integral(
    xs,
    priorValues.map((value, index) =>
      Math.min(value / priorArea, (posteriorValues[index] ?? 0) / posteriorArea),
    ),
  );
  return Math.max(0, Math.min(1, coefficient));
}

function interpolate(points, xs) {
  let upper = 1;
  return xs.map((x) => {
    if (
      points.length === 0 ||
      x < (points[0]?.x ?? 0) ||
      x > (points.at(-1)?.x ?? 0)
    ) {
      return 0;
    }
    if (points.length === 1) return points[0]?.density ?? 0;
    while (upper < points.length - 1 && (points[upper]?.x ?? x) < x) upper += 1;
    const left = points[upper - 1];
    const right = points[upper];
    if (left === undefined || right === undefined) return 0;
    return (
      left.density +
      ((right.density - left.density) * (x - left.x)) / (right.x - left.x)
    );
  });
}

function integral(xs, values) {
  let area = 0;
  for (let index = 1; index < xs.length; index += 1) {
    area +=
      (((values[index - 1] ?? 0) + (values[index] ?? 0)) *
        ((xs[index] ?? 0) - (xs[index - 1] ?? 0))) /
      2;
  }
  return area;
}

function grid(min, max) {
  return Array.from({ length: 128 }, (_, index) => min + ((max - min) * index) / 127);
}

function extent(values) {
  const min = minimum(values);
  const max = maximum(values);
  return min === max ? [min - 0.5, max + 0.5] : [min, max];
}

function minimum(values, initial = Infinity) {
  let result = initial;
  for (const value of values) result = Math.min(result, value);
  return result;
}

function maximum(values, initial = -Infinity) {
  let result = initial;
  for (const value of values) result = Math.max(result, value);
  return result;
}

function scale(min, max, start, end) {
  return (value) => start + ((value - min) / (max - min)) * (end - start);
}

function svg(width, height, label, content) {
  return `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(label)}" width="${String(width)}" height="${String(height)}" viewBox="0 0 ${String(width)} ${String(height)}" style="${FONT};background:var(--card)">${content}</svg>`;
}

function f(value) {
  return value.toFixed(2);
}

function path(points) {
  return points
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"}${f(point[0] ?? 0)} ${f(point[1] ?? 0)}`,
    )
    .join(" ");
}

function esc(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
