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
      assert(report.diagnostics_format === "v0-provisional", `unexpected report: ${report.diagnostics_format}`);
    },
  },
  {
    name: "prior predictive ignores sampler-only settings",
    fn: async () => {
      const runtime = new BrowserRuntime();
      const model = await (await fetch("/tests/fixtures/corpus/linear_regression.json")).text();
      const data = JSON.stringify({
        format: "bayescycle.data.json.v1",
        variables: { x: { dtype: "float64", shape: [3], values: [-1, 0, 1] } },
      });
      const result = await runtime.run({
        operation: "prior-predictive",
        modelIr: model,
        data,
        settings: { num_draws: 4, num_warmup: 10, max_treedepth: 8, target_accept: 0.9, seed: 3 },
      });
      assert(result.artifacts[0]?.name === "prior_predictive.ndjson", "prior artifact missing");
    },
  },
  {
    name: "runtime preserves exact model bytes and returns enveloped run inputs",
    fn: async () => {
      const modelIr = new TextEncoder().encode(' { "bayeswire_ir" : 1 }\n');
      const data = new TextEncoder().encode('{"format":"bayescycle.data.json.v1","variables":{}}\n');
      const requests = [];
      const executor = {
        execute: async (request) => {
          requests.push(request);
          return { rawBytes: new TextEncoder().encode('{"kind":"header"}\n{"x":1}\n{"kind":"trailer"}\n') };
        },
      };
      const runtime = new BrowserRuntime(executor);
      const result = await runtime.run({
        type: "run", id: "exact-1", operation: "sample", modelIr, data,
        settings: { chains: 1, num_warmup: 0, num_draws: 4, seed: 1, max_treedepth: 4, target_accept: 0.8 },
      });
      assert(requests[0].model instanceof Uint8Array, "engine request did not retain model bytes");
      assert(UTF8.decode(requests[0].model) === UTF8.decode(modelIr), "engine request changed exact model bytes");
      assert(result.type === "artifacts" && result.id === "exact-1", `result envelope is malformed: ${JSON.stringify(result)}`);
      const byName = Object.fromEntries(result.artifacts.map((artifact) => [artifact.name, artifact]));
      assert(UTF8.decode(byName["model.ir.json"].bytes) === UTF8.decode(modelIr), "model artifact changed bytes");
      assert(UTF8.decode(byName["data.json"].bytes) === UTF8.decode(data), "data artifact changed bytes");
      assert(byName["posterior.ndjson"] !== undefined, "posterior artifact missing");
    },
  },
  {
    name: "runtime serializes accepted object data into the sample artifact",
    fn: async () => {
      const data = {
        format: "bayescycle.data.json.v1",
        variables: { x: { dtype: "int64", shape: [], values: [1] } },
      };
      const executor = {
        execute: async () => ({ rawBytes: new TextEncoder().encode('{"kind":"header"}\n{"x":1}\n{"kind":"trailer"}\n') }),
      };
      const result = await new BrowserRuntime(executor).run({
        type: "run", id: "object-data", operation: "sample",
        modelIr: new TextEncoder().encode('{"bayeswire_ir":1}'), data,
        settings: { chains: 1, num_warmup: 0, num_draws: 4, seed: 1, max_treedepth: 4, target_accept: 0.8 },
      });
      const artifact = result.artifacts.find((entry) => entry.name === "data.json");
      assert(artifact !== undefined, "object data did not produce data.json");
      assert(JSON.parse(UTF8.decode(artifact.bytes)).variables.x.values[0] === 1, "object data artifact changed");
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
