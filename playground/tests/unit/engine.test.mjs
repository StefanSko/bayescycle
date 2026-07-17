import {
  InProcessEngine,
  MAX_POSTERIOR_RESPONSE_BYTES,
  diagnose,
  mergeChainFits,
  parseEngineMetadata,
  parseEngineOutput,
  sample,
} from "/site/src/engine/index.mjs";
import * as engine from "/site/src/engine/index.mjs";

const ENGINE_ROOT = "/site/vendor/bayesite/";
const FIXTURE_ROOT = "/tests/fixtures/engine/";
const MODELS = ["eight_schools_non_centered", "varying_intercepts_poisson"];
const UTF8 = new TextDecoder();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function fetchBytes(url) {
  const response = await fetch(url);
  assert(response.ok, `request for ${url} failed with HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function fetchText(url) {
  const response = await fetch(url);
  assert(response.ok, `request for ${url} failed with HTTP ${response.status}`);
  return response.text();
}

function bytesEqual(left, right) {
  return left.length === right.length && left.every((byte, index) => byte === right[index]);
}

let executorPromise;
function sharedExecutor() {
  executorPromise ??= (async () => {
    const [wasmBytes, metadataText] = await Promise.all([
      fetchBytes(`${ENGINE_ROOT}bayesite_core.wasm`),
      fetchText(`${ENGINE_ROOT}ENGINE.json`),
    ]);
    const metadata = parseEngineMetadata(JSON.parse(metadataText));
    return InProcessEngine.create(wasmBytes, metadata);
  })();
  return executorPromise;
}

let sampledPromise;
function sampledFits() {
  sampledPromise ??= (async () => {
    const [executor, modelText, dataText] = await Promise.all([
      sharedExecutor(),
      fetchText(`${FIXTURE_ROOT}eight_schools_non_centered/model.ir.json`),
      fetchText(`${FIXTURE_ROOT}eight_schools_non_centered/data.json`),
    ]);
    const streamed = new Map();
    const result = await sample({
      model: JSON.parse(modelText),
      data: JSON.parse(dataText),
      settings: {
        num_warmup: 4,
        num_draws: 4,
        max_treedepth: 4,
        target_accept: 0.8,
      },
      seed: 17,
      chains: 2,
      executor,
      onDrawBatch: ({ chainId, draws }) => {
        streamed.set(chainId, (streamed.get(chainId) ?? 0) + draws.length);
      },
    });
    return { result, streamed };
  })();
  return sampledPromise;
}

export default [
  {
    name: "golden diagnose is byte-identical",
    fn: async () => {
      const executor = await sharedExecutor();
      for (const modelName of MODELS) {
        const root = `${FIXTURE_ROOT}${modelName}/`;
        const [fit, golden] = await Promise.all([
          fetchText(`${root}posterior.ndjson`),
          fetchBytes(`${root}diagnostics.json`),
        ]);
        const output = await executor.execute({ command: "diagnose", fit });
        assert(
          bytesEqual(output.rawBytes, golden),
          `${modelName} diagnose bytes differ: actual=${output.rawBytes.length} expected=${golden.length}`,
        );
      }
    },
  },
  {
    name: "streams per_draw_v2 in batches",
    fn: async () => {
      await sharedExecutor();
      const goldenFit = await fetchBytes(
        `${FIXTURE_ROOT}eight_schools_non_centered/posterior.ndjson`,
      );
      const batchSizes = [];
      const output = parseEngineOutput(goldenFit, 0, true, ({ draws }) => {
        batchSizes.push(draws.length);
      });
      assert(
        output.header?.sample_stats_mode === "per_draw_v2",
        `unexpected sample_stats_mode: ${output.header?.sample_stats_mode}`,
      );
      assert(batchSizes.length > 1, `expected multiple batches, got ${batchSizes.length}`);
      assert(
        batchSizes.slice(0, -1).every((size) => size >= 64),
        `non-final batch smaller than 64: ${batchSizes.join(", ")}`,
      );
      const total = batchSizes.reduce((sum, size) => sum + size, 0);
      assert(total === 600, `expected 600 streamed draws, got ${total}`);
    },
  },
  {
    name: "sample yields one output per chain and streams draws",
    fn: async () => {
      const { result, streamed } = await sampledFits();
      assert(result.ok, `sample failed: ${JSON.stringify(result.error)}`);
      assert(result.outputs.length === 2, `expected 2 outputs, got ${result.outputs.length}`);
      assert(
        streamed.get(0) === 4 && streamed.get(1) === 4 && streamed.size === 2,
        `unexpected streamed draw counts: ${JSON.stringify(Object.fromEntries(streamed))}`,
      );
      result.outputs.forEach((output, chainId) => {
        assert(output.header !== undefined, `chain ${chainId} has no header`);
        assert(output.trailer !== undefined, `chain ${chainId} has no trailer`);
      });
    },
  },
  {
    name: "diagnose merges the two chain fits",
    fn: async () => {
      const executor = await sharedExecutor();
      const { result: sampled } = await sampledFits();
      assert(sampled.ok, `sample failed: ${JSON.stringify(sampled.error)}`);
      const result = await diagnose({
        fits: sampled.outputs.map((output) => UTF8.decode(output.rawBytes)),
        executor,
      });
      assert(result.ok, `diagnose failed: ${JSON.stringify(result.error)}`);
      assert(result.outputs.length === 1, `expected one output, got ${result.outputs.length}`);
      const report = JSON.parse(UTF8.decode(result.outputs[0].rawBytes));
      assert(
        report !== null && typeof report === "object" && !Array.isArray(report),
        "diagnose output is not a JSON report object",
      );
    },
  },
  {
    name: "diagnose bounds merged fits by default before executor dispatch",
    fn: async () => {
      const fit = (chain) => [
        { chain_count: 1, chain_order: [chain], draw_count: 1 },
        { chain, draw_index: 0, values: { alpha: chain } },
        {
          trailer: {
            chain_count: 1,
            chain_order: [chain],
            draw_count: 1,
            parameter_order: ["alpha"],
            chains: [{ chain, draw_count: 1 }],
          },
        },
      ].map((value) => JSON.stringify(value)).join("\n") + "\n";
      const nativeEncode = TextEncoder.prototype.encode;
      let executorCalls = 0;
      let result;
      try {
        // Report an oversized encoded line without allocating a 64 MiB fixture.
        TextEncoder.prototype.encode = function () {
          return { byteLength: MAX_POSTERIOR_RESPONSE_BYTES + 1 };
        };
        result = await diagnose({
          fits: [fit(0), fit(1)],
          executor: {
            execute: async () => {
              executorCalls += 1;
              return { rawBytes: new Uint8Array([1]) };
            },
          },
        });
      } finally {
        TextEncoder.prototype.encode = nativeEncode;
      }
      assert(!result.ok && result.error.error === "PosteriorTooLarge",
        `default diagnose merge was not bounded: ${JSON.stringify(result)}`);
      assert(executorCalls === 0, `oversized diagnose merge reached executor ${executorCalls} times`);
    },
  },
  {
    name: "merged fits never retain first-chain diagnostics",
    fn: async () => {
      const fit = (chain, rhat, ess) => [
        { chain_count: 1, chain_order: [chain], draw_count: 1 },
        { chain, draw_index: 0, values: { alpha: chain } },
        {
          trailer: {
            chain_count: 1,
            chain_order: [chain],
            draw_count: 1,
            parameter_order: ["alpha"],
            chains: [{ chain, draw_count: 1 }],
            rhat: { alpha: rhat },
            ess: { alpha: ess },
          },
        },
      ].map((value) => JSON.stringify(value)).join("\n") + "\n";

      const merged = mergeChainFits([
        fit(0, 0.91, 101),
        fit(1, 1.09, 202),
      ]).trimEnd().split("\n").map((line) => JSON.parse(line));
      const trailer = merged.at(-1)?.trailer;
      assert(trailer?.chain_count === 2, `unexpected chain count: ${trailer?.chain_count}`);
      assert(trailer?.draw_count === 2, `unexpected draw count: ${trailer?.draw_count}`);
      assert(trailer?.chains.length === 2, `unexpected chain facts: ${trailer?.chains.length}`);
      assert(trailer?.rhat.alpha === null, `retained chain R-hat: ${trailer?.rhat.alpha}`);
      assert(trailer?.ess.alpha === null, `retained chain ESS: ${trailer?.ess.alpha}`);
    },
  },
  {
    name: "merged fit serialization enforces its byte ceiling",
    fn: () => {
      const fit = (chain) => [
        { chain_count: 1, chain_order: [chain], draw_count: 1 },
        { chain, draw_index: 0, values: { alpha: chain } },
        {
          trailer: {
            chain_count: 1,
            chain_order: [chain],
            draw_count: 1,
            parameter_order: ["alpha"],
            chains: [{ chain, draw_count: 1 }],
          },
        },
      ].map((value) => JSON.stringify(value)).join("\n") + "\n";
      const fits = [fit(0), fit(1)];
      const merged = mergeChainFits(fits);
      const exactBytes = new TextEncoder().encode(merged).byteLength;
      assert(mergeChainFits(fits, exactBytes) === merged, "exact merged ceiling was rejected");
      let error;
      try {
        mergeChainFits(fits, exactBytes - 1);
      } catch (reason) {
        error = reason;
      }
      assert(error?.error === "PosteriorTooLarge", `oversized merge was not typed: ${String(error)}`);
    },
  },
  {
    name: "malformed IR returns a typed engine error",
    fn: async () => {
      const [executor, dataText] = await Promise.all([
        sharedExecutor(),
        fetchText(`${FIXTURE_ROOT}eight_schools_non_centered/data.json`),
      ]);
      const result = await sample({
        model: { bayeswire_ir: 1 },
        data: JSON.parse(dataText),
        settings: {
          num_warmup: 4,
          num_draws: 4,
          max_treedepth: 4,
          target_accept: 0.8,
        },
        seed: 1,
        chains: 1,
        executor,
      });
      assert(!result.ok, "malformed IR unexpectedly sampled successfully");
      assert(
        result.error.error_format === "v0-provisional",
        `unexpected error format: ${result.error.error_format}`,
      );
      assert(
        result.error.error === "MalformedDocument",
        `unexpected typed error: ${result.error.error}`,
      );
    },
  },
  {
    name: "exports the six retained verbs including native generation",
    fn: async () => {
      await sharedExecutor();
      const verbs = [
        engine.sample,
        engine.diagnose,
        engine.generate,
        engine.recover,
        engine.recoverCheck,
        engine.sbc,
      ];
      assert(verbs.every((verb) => typeof verb === "function"), "not all six verbs are functions");
      const retired = ["priorPredictive", "posteriorPredictive", "posteriorCheck", "simulate"];
      assert(
        retired.every((verb) => engine[verb] === undefined),
        "a retired legacy verb is still exported",
      );
    },
  },
];
