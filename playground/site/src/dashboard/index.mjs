import { renderEssRhat, renderPrecis, renderTrank } from "./render.mjs";

export * from "./data.mjs";
export * from "./render.mjs";
export * from "./stats.mjs";

export const MAX_DASHBOARD_PARAMETER_COMPONENTS = 200;

export function dashboardPlotLimit(data) {
  const componentCount = data.parameters.length;
  return componentCount > MAX_DASHBOARD_PARAMETER_COMPONENTS
    ? {
        kind: "omitted",
        componentCount,
        notice: `Posterior has ${componentCount} parameter components; plots are shown for models with at most ${MAX_DASHBOARD_PARAMETER_COMPONENTS}.`,
      }
    : { kind: "allowed", componentCount };
}

export function prepareDashboardPlots(data, truth) {
  const limit = dashboardPlotLimit(data);
  if (limit.kind === "omitted") return limit;
  return {
    kind: "rendered",
    componentCount: limit.componentCount,
    trank: renderTrank(data),
    essRhat: renderEssRhat(data),
    precis: renderPrecis(data, truth),
  };
}
