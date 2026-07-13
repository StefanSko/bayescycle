// Source: src/dashboard/stats.ts, bayesledger @ 7346d71.

/** Gaussian KDE on the frozen 128-point, data ± 3h grid. */
export function gaussianKde(values) {
  if (values.length === 0) return [];
  const mean = average(values);
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const deviation = Math.sqrt(variance);
  const bandwidth =
    deviation > 0
      ? 1.06 * deviation * values.length ** -0.2
      : Math.max(Math.abs(mean) * 0.01, 0.01);
  const start = Math.min(...values) - 3 * bandwidth;
  const end = Math.max(...values) + 3 * bandwidth;
  const normalizer = values.length * bandwidth * Math.sqrt(2 * Math.PI);
  return Array.from({ length: 128 }, (_, index) => {
    const x = start + ((end - start) * index) / 127;
    const density =
      values.reduce(
        (sum, value) => sum + Math.exp(-0.5 * ((x - value) / bandwidth) ** 2),
        0,
      ) / normalizer;
    return { x, density };
  });
}

/** Pooled average ranks (ties share a rank), split into 20 equal rank bins per chain. */
export function rankHistogram(chains, bins = 20) {
  const pooled = chains.flatMap((values, chain) =>
    values.map((value, index) => ({ value, chain, index })),
  );
  const sorted = [...pooled].sort(
    (left, right) =>
      left.value - right.value || left.chain - right.chain || left.index - right.index,
  );
  const ranks = new Map();
  let start = 0;
  while (start < sorted.length) {
    let end = start + 1;
    while (end < sorted.length && sorted[end]?.value === sorted[start]?.value) end += 1;
    const rank = (start + 1 + end) / 2;
    for (let index = start; index < end; index += 1) {
      const item = sorted[index];
      if (item !== undefined) ranks.set(`${String(item.chain)}:${String(item.index)}`, rank);
    }
    start = end;
  }
  return chains.map((values, chain) => {
    const counts = Array.from({ length: bins }, () => 0);
    for (const [index] of values.entries()) {
      const rank = ranks.get(`${String(chain)}:${String(index)}`) ?? 1;
      const bin = Math.min(
        bins - 1,
        Math.floor(((rank - 1) * bins) / Math.max(pooled.length, 1)),
      );
      counts[bin] = (counts[bin] ?? 0) + 1;
    }
    return counts;
  });
}

export function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function quantile(values, probability) {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 0) throw new Error("Quantile needs values");
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const low = sorted[lower] ?? 0;
  const high = sorted[upper] ?? low;
  return low + (high - low) * (position - lower);
}
