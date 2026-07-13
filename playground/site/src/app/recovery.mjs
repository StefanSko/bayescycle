export function renderRecoverySummary(report) {
  if (report === null || typeof report !== "object" || !Array.isArray(report.target_order) ||
      report.targets === null || typeof report.targets !== "object") {
    throw new Error("Recovery report is malformed");
  }
  const percentage = Number.isFinite(report.interval) ? Math.round(report.interval * 100) : 0;
  const rows = report.target_order.map((name) => {
    const target = report.targets[name];
    if (target === null || typeof target !== "object") throw new Error(`Recovery target ${name} is malformed`);
    const verdict = target.interval_contains_truth === true ? "inside" : "outside";
    return `<tr><th scope="row">${escapeHtml(name)}</th><td>${format(target.truth)}</td><td>${format(target.mean)}</td><td>${format(target.lower)}–${format(target.upper)}</td><td class="recovery-${verdict}">${verdict}</td></tr>`;
  }).join("");
  return `<p>${percentage}% interval recovery facts from Bayesite.</p><table><thead><tr><th>Target</th><th>Truth</th><th>Mean</th><th>${percentage}% interval</th><th>Truth</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function format(value) {
  if (Array.isArray(value)) return escapeHtml(JSON.stringify(value));
  return typeof value === "number" && Number.isFinite(value) ? String(Number(value.toPrecision(4))) : escapeHtml(String(value));
}

function escapeHtml(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}
