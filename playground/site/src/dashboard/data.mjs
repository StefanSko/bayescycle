// Source: src/dashboard/data.ts, bayesledger @ 7346d71.

import { parseEngineOutput } from "../engine/stream.mjs";

const UTF8 = new TextEncoder();
const DECODE = new TextDecoder();

/**
 * @param {{fits: (string|Uint8Array)[], diagnose: object|string|Uint8Array}} input
 */
export function readDashboardData(input) {
  if (input.fits.length === 0) throw new Error("Dashboard needs at least one fit stream");
  const draws = [];
  let header;
  for (const [fitIndex, fit] of input.fits.entries()) {
    const output = parseEngineOutput(bytes(fit), fitIndex, true, (batch) => {
      draws.push(...batch.draws);
    });
    header ??= output.header;
  }
  if (header === undefined) throw new Error("Fit stream has no header");
  const report = reportObject(input.diagnose);
  const specs = paramSpecs(header);
  const chainIds = [
    ...new Set(draws.map((draw) => integer(draw.chain, "draw chain"))),
  ].sort((a, b) => a - b);
  const parameters = specs.flatMap((spec) =>
    spec.coordinates.map((coordinate, coordinateIndex) => ({
      name: spec.name,
      label: coordinate.length === 0 ? spec.name : `${spec.name}[${coordinate.join(",")}]`,
      coordinate,
      chains: chainIds.map((chain) =>
        draws
          .filter((draw) => draw.chain === chain)
          .map((draw) => drawValue(draw, spec.name, coordinateIndex)),
      ),
      rhat: metric(report.rhat, spec.name, coordinateIndex),
      ess: metric(report.ess, spec.name, coordinateIndex),
    })),
  );
  const reportParameters = specs.map((spec) => ({
    name: spec.name,
    rhat: metric(report.rhat, spec.name, 0),
    ess: metric(report.ess, spec.name, 0),
  }));
  return {
    parameters,
    reportParameters,
    energies: chainIds.map((chain) =>
      draws
        .filter((draw) => draw.chain === chain)
        .map((draw) => finite(draw.energy, "energy")),
    ),
    divergences: draws.filter((draw) => draw.diverging === true).length,
    drawCount: draws.length,
    chainCount: chainIds.length,
  };
}

function bytes(value) {
  return typeof value === "string" ? UTF8.encode(value) : value;
}

function reportObject(value) {
  if (typeof value !== "string" && !(value instanceof Uint8Array)) return value;
  const parsed = JSON.parse(typeof value === "string" ? value : DECODE.decode(value));
  if (!object(parsed)) throw new Error("Diagnose report must be an object");
  return parsed;
}

function paramSpecs(header) {
  const params = header.params;
  if (!Array.isArray(params)) throw new Error("Fit header params must be an array");
  return params.map((value) => {
    if (!object(value) || typeof value.name !== "string") {
      throw new Error("Invalid parameter metadata");
    }
    const order = value.coordinate_order;
    const coordinates = Array.isArray(order)
      ? order.map((coordinate) => {
          if (!Array.isArray(coordinate) || !coordinate.every(Number.isInteger)) {
            throw new Error("Invalid coordinate_order");
          }
          return coordinate;
        })
      : [[]];
    return { name: value.name, coordinates };
  });
}

function drawValue(draw, name, coordinateIndex) {
  const values = draw.values;
  if (!object(values)) throw new Error("Draw values must be an object");
  const value = values[name];
  return Array.isArray(value)
    ? finite(value[coordinateIndex], `${name} coordinate`)
    : finite(value, name);
}

function metric(container, name, coordinateIndex) {
  if (!object(container)) return null;
  const value = container[name];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (Array.isArray(value)) {
    const item = value[coordinateIndex];
    return typeof item === "number" && Number.isFinite(item) ? item : null;
  }
  return null;
}

function object(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finite(value, context) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid ${context}`);
  }
  return value;
}

function integer(value, context) {
  if (!Number.isInteger(value)) throw new Error(`Invalid ${context}`);
  return value;
}
