// Source: src/dashboard/render.ts, bayesledger @ 7346d71.

import { average, gaussianKde, quantile, rankHistogram } from "./stats.mjs";

const FONT = "font-family:var(--mono);font-size:12px";
const axis = "var(--line-strong)";
const ink = "var(--ink)";
const accent = "var(--accent)";
const danger = "var(--danger)";
const muted = "var(--muted)";

export function renderEssRhat(data) {
  const width = 700;
  const left = 58;
  const right = 300;
  const top = 38;
  const labelX = 330;
  const rowGap = 18;
  const available = data.reportParameters.filter(
    (parameter) => parameter.ess !== null && parameter.rhat !== null,
  );
  const missing = data.reportParameters
    .filter((parameter) => parameter.ess === null || parameter.rhat === null)
    .sort((a, b) => compare(a.name, b.name));
  const height = Math.max(260, 78 + (available.length + missing.length) * rowGap);
  const bottom = height - 50;
  const maxEss = maximum(
    available.map((parameter) => parameter.ess ?? 0),
    Math.max(data.drawCount, 1),
  );
  const rhats = available.map((parameter) => parameter.rhat ?? 1);
  const minRhat = minimum(rhats, 0.98);
  const maxRhat = maximum(rhats, 1.05);
  const x = (value) => left + (value / maxEss) * (right - left);
  const y = (value) =>
    bottom - ((value - minRhat) / (maxRhat - minRhat)) * (bottom - top);
  const sorted = [...available].sort(
    (a, b) => y(a.rhat ?? 1) - y(b.rhat ?? 1) || compare(a.name, b.name),
  );
  const guide = x(data.drawCount * 0.1);
  const labels = sorted
    .map((parameter, index) => {
      const pointX = x(parameter.ess ?? 0);
      const pointY = y(parameter.rhat ?? 1);
      const labelY = 60 + index * rowGap;
      return `<circle cx="${f(pointX)}" cy="${f(pointY)}" r="4.00" fill="${accent}"/><path d="M${f(pointX + 5)} ${f(pointY)} L${f(labelX - 8)} ${f(labelY - 4)}" fill="none" stroke="${muted}"/><text x="${f(labelX)}" y="${f(labelY)}" fill="${ink}">${escape(parameter.name)} · ESS ${String(Math.round(parameter.ess ?? 0))} · R-hat ${(parameter.rhat ?? 0).toFixed(3)}</text>`;
    })
    .join("");
  const missingTop = 60 + sorted.length * rowGap;
  return svg(
    width,
    height,
    "ESS × R-hat",
    `<text x="18.00" y="22.00" fill="${ink}">ESS × R-hat</text><line x1="${f(left)}" y1="${f(bottom)}" x2="${f(right)}" y2="${f(bottom)}" stroke="${axis}"/><line x1="${f(left)}" y1="${f(top)}" x2="${f(left)}" y2="${f(bottom)}" stroke="${axis}"/><line x1="${f(guide)}" y1="${f(top)}" x2="${f(guide)}" y2="${f(bottom)}" stroke="${danger}"/><text x="${f(guide + 4)}" y="${f(top + 12)}" fill="${danger}">0.1 N = ${String(Math.round(data.drawCount * 0.1))}</text>${labels}<text x="${f(left)}" y="${f(bottom + 22)}" fill="${muted}">ESS</text><text x="8.00" y="${f(top)}" fill="${muted}">R-hat</text>${missing.length === 0 ? "" : `<text x="${f(labelX)}" y="${f(missingTop)}" fill="${muted}">not available</text>${missing.map((parameter, index) => `<text x="${f(labelX)}" y="${f(missingTop + (index + 1) * rowGap)}" fill="${danger}">${escape(parameter.name)}</text>`).join("")}`}`,
  );
}

export function renderEnergy(data) {
  const width = 540;
  const row = 145;
  const height = 40 + data.energies.length * row;
  const panels = data.energies
    .map((values, chain) => {
      const kde = gaussianKde(values);
      if (kde.length === 0) return "";
      const mean = average(values);
      const variance =
        values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
      const sd = Math.sqrt(variance) || 0.01;
      const minX = kde[0]?.x ?? 0;
      const maxX = kde.at(-1)?.x ?? 1;
      const normal = kde.map(
        ({ x }) =>
          Math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI)),
      );
      const maxY = maximum(normal, maximum(kde.map((point) => point.density), 0.001));
      const y0 = 55 + chain * row;
      const plotBottom = y0 + 100;
      const px = (value) => 58 + ((value - minX) / (maxX - minX)) * 450;
      const py = (value) => plotBottom - (value / maxY) * 88;
      return `<text x="18.00" y="${f(y0)}" fill="${ink}">chain ${String(chain + 1)}</text><line x1="58.00" y1="${f(plotBottom)}" x2="508.00" y2="${f(plotBottom)}" stroke="${axis}"/><path d="${path(kde.map((point) => [px(point.x), py(point.density)]))}" fill="none" stroke="${accent}" stroke-width="2"/><path d="${path(kde.map((point, index) => [px(point.x), py(normal[index] ?? 0)]))}" fill="none" stroke="${danger}" stroke-dasharray="5 4"/><text x="58.00" y="${f(plotBottom + 17)}" fill="${muted}">${minX.toPrecision(3)}</text><text x="470.00" y="${f(plotBottom + 17)}" fill="${muted}">${maxX.toPrecision(3)}</text>`;
    })
    .join("");
  return svg(
    width,
    height,
    "HMC energy",
    `<text x="18.00" y="22.00" fill="${ink}">HMC energy · density / moment-matched normal</text>${panels}`,
  );
}

export function divergenceVerdict(count) {
  return count === 0
    ? "outlook good"
    : count < 10
      ? "roll again"
      : "check yourself before you wreck yourself";
}

export function renderDivergences(data) {
  const verdict = divergenceVerdict(data.divergences);
  const color = data.divergences === 0 ? accent : danger;
  return svg(
    540,
    130,
    "Divergence verdict",
    `<text x="18.00" y="25.00" fill="${ink}">divergent transitions</text><text x="18.00" y="70.00" fill="${color}" font-size="30px">${String(data.divergences)} / ${String(data.drawCount)}</text><text x="18.00" y="105.00" fill="${color}">${verdict}</text>`,
  );
}

export function renderTrank(data, mode = "rank") {
  const panelHeight = 128;
  const height = 35 + data.parameters.length * panelHeight;
  const panels = data.parameters
    .map((parameter, panelIndex) => {
      const top = 35 + panelIndex * panelHeight;
      const bottom = top + 88;
      const label = `<text x="18.00" y="${f(top + 10)}" fill="${ink}">${escape(parameter.label)} · n_eff ${parameter.ess === null ? "—" : String(Math.round(parameter.ess))}</text>`;
      if (mode === "trace") {
        const pooled = parameter.chains.flat();
        const min = minimum(pooled);
        const max = maximum(pooled);
        const span = max - min || 1;
        const maxLength = maximum(parameter.chains.map((chain) => chain.length));
        return `${label}<line x1="58.00" y1="${f(bottom)}" x2="510.00" y2="${f(bottom)}" stroke="${axis}"/>${parameter.chains.map((chain, chainIndex) => `<path d="${path(chain.map((value, draw) => [58 + (draw / Math.max(maxLength - 1, 1)) * 452, bottom - ((value - min) / span) * 66]))}" fill="none" stroke="${chainIndex === 0 ? accent : danger}" opacity="0.80"/>`).join("")}`;
      }
      const histograms = rankHistogram(parameter.chains);
      const maxCount = maximum(histograms.flat(), 1);
      const binWidth = 452 / 20;
      return `${label}<line x1="58.00" y1="${f(bottom)}" x2="510.00" y2="${f(bottom)}" stroke="${axis}"/>${histograms.map((counts, chain) => counts.map((count, bin) => `<rect x="${f(58 + bin * binWidth + (chain * binWidth) / data.chainCount)}" y="${f(bottom - (count / maxCount) * 66)}" width="${f(binWidth / data.chainCount)}" height="${f((count / maxCount) * 66)}" fill="${chain === 0 ? accent : danger}" opacity="0.72"/>`).join("")).join("")}`;
    })
    .join("");
  return svg(
    540,
    height,
    mode === "rank" ? "Trank grid" : "Trace grid",
    `<text x="18.00" y="20.00" fill="${muted}">${mode === "rank" ? "pooled ranks · 20 bins" : "retained draws only"}</text>${panels}`,
  );
}

export function renderPrecis(data, truth) {
  const rows = data.parameters
    .map((parameter) => ({ parameter, values: parameter.chains.flat() }))
    .filter((row) => row.values.length > 0);
  const summaries = rows.map(({ parameter, values }) => ({
    label: parameter.label,
    mean: average(values),
    low: quantile(values, 0.055),
    high: quantile(values, 0.945),
  }));
  const truthValues =
    truth === undefined
      ? []
      : summaries
          .filter((summary) => Object.hasOwn(truth, summary.label))
          .map((summary) => truth[summary.label]);
  const minimumValue = minimum(
    truthValues,
    minimum(summaries.map((summary) => summary.low), 0),
  );
  const maximumValue = maximum(
    truthValues,
    maximum(summaries.map((summary) => summary.high), 0),
  );
  const span = maximumValue - minimumValue || 1;
  const plotLeft = 150;
  const plotRight = 500;
  const x = (value) => plotLeft + ((value - minimumValue) / span) * (plotRight - plotLeft);
  const axisY = 60 + summaries.length * 28;
  const legendY = axisY + 38;
  const height = legendY + 18;
  const ticks = [minimumValue];
  if (minimumValue < 0 && maximumValue > 0) ticks.push(0);
  if (maximumValue !== minimumValue) ticks.push(maximumValue);
  const hasTruth = truthValues.length > 0;
  const tickMarkup = ticks.map((value) =>
    `<line class="value-axis-tick" x1="${f(x(value))}" y1="${f(axisY)}" x2="${f(x(value))}" y2="${f(axisY + 5)}" stroke="${axis}"/><text class="value-axis-label" x="${f(x(value))}" y="${f(axisY + 18)}" fill="${muted}" text-anchor="middle">${axisLabel(value)}</text>`).join("");
  const legend = `<g class="precis-legend"><line x1="18.00" y1="${f(legendY)}" x2="38.00" y2="${f(legendY)}" stroke="${accent}" stroke-width="3"/><circle cx="28.00" cy="${f(legendY)}" r="3.00" fill="${accent}"/><text x="44.00" y="${f(legendY + 4)}" fill="${muted}">posterior 89% interval · mean</text>${hasTruth ? `<line x1="300.00" y1="${f(legendY - 7)}" x2="300.00" y2="${f(legendY + 7)}" stroke="${danger}" stroke-width="2"/><text x="310.00" y="${f(legendY + 4)}" fill="${muted}">true value</text>` : ""}</g>`;
  return svg(
    540,
    height,
    "Precis",
    `<text x="18.00" y="22.00" fill="${ink}">Precis · mean and 89% interval</text>${summaries.map((summary, index) => {
      const y = 52 + index * 28;
      const hasParameterTruth = truth !== undefined && Object.hasOwn(truth, summary.label);
      const marker = hasParameterTruth
        ? `<line class="truth-marker" data-parameter="${escape(summary.label)}" x1="${f(x(truth[summary.label]))}" y1="${f(y - 8)}" x2="${f(x(truth[summary.label]))}" y2="${f(y + 8)}" stroke="${danger}" stroke-width="2"/>`
        : "";
      const annotation = hasParameterTruth
        ? `<tspan fill="${danger}"> · true ${truth[summary.label].toPrecision(3)}</tspan>`
        : "";
      return `<text x="18.00" y="${f(y + 4)}" fill="${ink}">${escape(summary.label)}</text><line x1="${f(x(summary.low))}" y1="${f(y)}" x2="${f(x(summary.high))}" y2="${f(y)}" stroke="${accent}" stroke-width="2"/><circle cx="${f(x(summary.mean))}" cy="${f(y)}" r="4.00" fill="${accent}"/><text x="${f(x(summary.high) + 5)}" y="${f(y + 4)}" fill="${muted}">${summary.mean.toPrecision(3)} [${summary.low.toPrecision(3)}, ${summary.high.toPrecision(3)}]${annotation}</text>${marker}`;
    }).join("")}<line class="value-axis" x1="${f(plotLeft)}" y1="${f(axisY)}" x2="${f(plotRight)}" y2="${f(axisY)}" stroke="${axis}"/>${tickMarkup}${legend}`,
  );
}

function axisLabel(value) {
  const normalized = Object.is(value, -0) ? 0 : value;
  return String(Number(normalized.toPrecision(3)));
}

function svg(width, height, label, content) {
  return `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${escape(label)}" width="${String(width)}" height="${String(height)}" viewBox="0 0 ${String(width)} ${String(height)}" style="${FONT};background:var(--card)">${content}</svg>`;
}

function f(value) {
  return value.toFixed(2);
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

function path(points) {
  return points
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${f(x)} ${f(y)}`)
    .join(" ");
}

function escape(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function compare(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}
