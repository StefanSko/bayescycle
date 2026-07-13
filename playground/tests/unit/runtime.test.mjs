import { BrowserRuntime } from "/site/src/runtime/browser-runtime.mjs";

const FIXTURE = "/tests/fixtures/engine/eight_schools_non_centered/";
const UTF8 = new TextDecoder();

function assert(condition, message) { if (!condition) throw new Error(message); }
async function text(name) { const response = await fetch(`${FIXTURE}${name}`); return response.text(); }

export default [
  {
    name: "runtime returns named diagnostic artifacts",
    fn: async () => {
      const runtime = new BrowserRuntime();
      const result = await runtime.run({ operation: "diagnose", fit: await text("posterior.ndjson") });
      assert(result.artifacts.length === 1, `unexpected artifacts: ${result.artifacts.length}`);
      assert(result.artifacts[0].name === "diagnostics.json", `unexpected name: ${result.artifacts[0].name}`);
      const report = JSON.parse(UTF8.decode(result.artifacts[0].bytes));
      assert(report.format === "bayesite.diagnostics.v0", `unexpected report: ${report.format}`);
    },
  },
  {
    name: "runtime sampling returns valid merged posterior and progress",
    fn: async () => {
      const runtime = new BrowserRuntime();
      const progress = [];
      const result = await runtime.run({
        operation: "sample",
        modelIr: await text("model.ir.json"), data: await text("data.json"),
        settings: { chains: 2, num_warmup: 4, num_draws: 4, seed: 17, max_treedepth: 4, target_accept: 0.8 },
      }, (event) => progress.push(event));
      const posterior = result.artifacts.find((artifact) => artifact.name === "posterior.ndjson");
      assert(posterior !== undefined, "posterior artifact missing");
      const lines = UTF8.decode(posterior.bytes).trimEnd().split("\n");
      assert(lines.length === 10, `expected header + 8 draws + trailer, got ${lines.length}`);
      assert(lines.every((line) => line.length > 0 && JSON.parse(line)), "invalid NDJSON line");
      assert(progress.some((event) => event.chainId === 0) && progress.some((event) => event.chainId === 1), "chain progress missing");
    },
  },
];
