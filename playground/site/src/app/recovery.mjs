export function renderRecoverySummary(report) {
  requireRecoveryReport(report);
  const percentage = Number.isFinite(report.interval) ? Math.round(report.interval * 100) : 0;
  const rows = report.target_order.map((name) => {
    const target = report.targets[name];
    if (target === null || typeof target !== "object") throw new Error(`Recovery target ${name} is malformed`);
    const verdict = target.interval_contains_truth === true ? "inside" : "outside";
    return `<tr><th scope="row">${escapeHtml(name)}</th><td>${format(target.truth)}</td><td>${format(target.mean)}</td><td>${format(target.lower)}–${format(target.upper)}</td><td class="recovery-${verdict}">${verdict}</td></tr>`;
  }).join("");
  return `<p>${percentage}% interval recovery facts from Bayesite.</p><table><thead><tr><th>Target</th><th>Truth</th><th>Mean</th><th>${percentage}% interval</th><th>Truth</th></tr></thead><tbody>${rows}</tbody></table>`;
}

export function recoveryTruthMap(report, parameterLabels = []) {
  requireRecoveryReport(report);
  if (!Array.isArray(parameterLabels) ||
      !parameterLabels.every((label) => typeof label === "string")) {
    throw new Error("Recovery parameter labels are malformed");
  }
  const labelsByParameter = new Map();
  for (const label of parameterLabels) {
    const bracket = label.indexOf("[");
    const name = bracket === -1 ? label : label.slice(0, bracket);
    const labels = labelsByParameter.get(name);
    if (labels === undefined) labelsByParameter.set(name, [label]);
    else labels.push(label);
  }
  const truth = {};
  for (const name of report.target_order) {
    const target = report.targets[name];
    if (target === null || typeof target !== "object") {
      throw new Error(`Recovery target ${String(name)} is malformed`);
    }
    const labels = labelsByParameter.get(name) ?? [];
    const values = flattenedTruth(target.truth, name);
    if (labels.length === values.length && labels.length > 0) {
      for (const [index, label] of labels.entries()) defineTruth(truth, label, values[index]);
    } else {
      addTruthValues(truth, name, target.truth, []);
    }
  }
  return truth;
}

function addTruthValues(output, name, value, coordinate) {
  if (Array.isArray(value)) {
    for (const [index, entry] of value.entries()) {
      addTruthValues(output, name, entry, [...coordinate, index]);
    }
    return;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Recovery truth for ${String(name)} is malformed`);
  }
  const label = coordinate.length === 0 ? name : `${name}[${coordinate.join(",")}]`;
  defineTruth(output, label, value);
}

function flattenedTruth(value, name) {
  const values = [];
  const visit = (entry) => {
    if (Array.isArray(entry)) {
      for (const child of entry) visit(child);
    } else if (typeof entry === "number" && Number.isFinite(entry)) {
      values.push(entry);
    } else {
      throw new Error(`Recovery truth for ${String(name)} is malformed`);
    }
  };
  visit(value);
  return values;
}

function defineTruth(output, label, value) {
  Object.defineProperty(output, label, {
    value,
    enumerable: true,
    writable: true,
    configurable: true,
  });
}

function requireRecoveryReport(report) {
  if (report === null || typeof report !== "object" || !Array.isArray(report.target_order) ||
      !report.target_order.every((name) => typeof name === "string") ||
      report.targets === null || typeof report.targets !== "object" ||
      Array.isArray(report.targets)) {
    throw new Error("Recovery report is malformed");
  }
}

function format(value) {
  if (Array.isArray(value)) return escapeHtml(JSON.stringify(value));
  return typeof value === "number" && Number.isFinite(value) ? String(Number(value.toPrecision(4))) : escapeHtml(String(value));
}

function escapeHtml(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}
