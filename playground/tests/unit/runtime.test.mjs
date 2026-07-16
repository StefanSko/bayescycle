import { BrowserRuntime } from "/site/src/runtime/browser-runtime.mjs";
import { EngineError } from "/site/src/engine/types.mjs";
import {
  fitArtifact,
  fixed,
  generateDatasets,
  modelPrior,
  posteriorOf,
} from "/site/src/generation/plan.mjs";

const FIXTURE = "/tests/fixtures/engine/eight_schools_non_centered/";
const UTF8 = new TextDecoder();

function assert(condition, message) { if (!condition) throw new Error(message); }
async function text(name) { const response = await fetch(`${FIXTURE}${name}`); return response.text(); }
function bytes(value) { return new TextEncoder().encode(value); }
async function hash(value) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", value));
  return `sha256:${[...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function runtimePosteriorBytes() {
  const workflowPhases = [
    "parse_json", "decode_ir", "bind_data", "build_posterior_state",
    "evaluate_logp_grad", "run_nuts", "emit_artifact",
  ];
  return bytes([
    {
      draws_format: "v0-provisional",
      artifact_kind: "posterior_draws",
      artifact_scope: "observed_data_conditioned_parameter_draws",
      workflow_phases: workflowPhases,
      params: [{ name: "theta", shape: [], coordinate_order: [[]] }],
      parameter_count: 1,
      parameter_order: ["theta"],
      packing: ["theta"],
      settings: { num_warmup: 0, num_draws: 1, max_treedepth: 4, target_accept: 0.8 },
      seed: 0,
      chain_count: 1,
      chains: 1,
      chain_order: [0],
      draw_count: 1,
      sample_stats_mode: "per_draw_v2",
    },
    {
      draws_format: "v0-provisional",
      artifact_kind: "posterior_draws",
      artifact_scope: "observed_data_conditioned_parameter_draws",
      draw_index: 0, draw_index_base: "zero_based_retained_draw_order",
      seed: 0, draw_count: 1, chain_count: 1, chain_order: [0],
      chain: 0, draw: 0, sample_stats_mode: "per_draw_v2",
      tree_depth: 1, tree_accept: 0.9, energy: 1.0, diverging: false, parameter_count: 1,
      parameter_order: ["theta"], values: { theta: 0.5 },
    },
    {
      trailer: {
        draws_format: "v0-provisional",
        artifact_kind: "posterior_draws",
        artifact_scope: "observed_data_conditioned_parameter_draws",
        workflow_phases: workflowPhases,
        seed: 0,
        draws_per_chain: 1,
        chain_count: 1,
        chain_order: [0],
        draw_count: 1,
        parameter_count: 1,
        parameter_order: ["theta"],
        params: 1,
        chains: [{
          chain: 0, draw_count: 1, divergences: 0,
          treedepth_histogram: [0, 1, 0, 0, 0], step_size: 0.1, mean_accept: 0.9,
        }],
      },
    },
  ].map((document) => JSON.stringify(document)).join("\n") + "\n");
}

function generatedOutput(request) {
  const marker = {
    generated_datasets_format: "v0-provisional",
    artifact_kind: "generated_dataset_pairs",
    artifact_scope: "parameter_and_complete_dataset_joint_draws",
  };
  const phases = [
    "parse_json", "decode_ir", "bind_design", "draw_parameters",
    "simulate_outcomes", "emit_artifact",
  ];
  const parameters = request.parameter_source.kind === "fixed"
    ? JSON.parse(request.parameter_source.parameters)
    : {
        format: "bayescycle.data.json.v1",
        variables: { theta: { dtype: "float64", shape: [], values: [0.5] } },
      };
  const dataset = JSON.parse(request.design);
  const schema = (document) => Object.entries(document.variables).map(([name, value]) => ({
    name, dtype: value.dtype, shape: value.shape,
  }));
  const source = request.parameter_source.kind === "fixed"
    ? { kind: "fixed", parameters_hash: request.identities.parameters_hash }
    : request.parameter_source.kind === "model-prior"
      ? {
          kind: "model-prior",
          model_hash: request.identities.generation_model_hash,
          authored_provenance: null,
        }
      : {
          kind: "posterior",
          fit_hash: request.identities.fit_hash,
          fit_model_hash: request.identities.fit_model_hash,
          fit_data_hash: request.identities.fit_data_hash,
        };
  const header = {
    ...marker, workflow_phases: phases,
    generation_model_hash: request.identities.generation_model_hash,
    design_hash: request.identities.design_hash,
    parameter_source: source,
    count: request.count,
    seed: request.seed,
    draw_index_base: "zero_based_generation_order",
    parameter_schema: schema(parameters),
    dataset_schema: schema(dataset),
  };
  const draws = Array.from({ length: request.count }, (_, drawIndex) => ({
    ...marker,
    draw_index: drawIndex,
    draw_count: request.count,
    parameters,
    dataset,
    source_lineage: request.parameter_source.kind === "fixed"
      ? { kind: "fixed" }
      : request.parameter_source.kind === "model-prior"
        ? { kind: "model-prior", source_draw_index: drawIndex }
        : { kind: "posterior", source_draw_index: 0, chain: 0, draw: 0 },
  }));
  const trailer = {
    ...marker, workflow_phases: phases,
    generation_model_hash: request.identities.generation_model_hash,
    design_hash: request.identities.design_hash,
    parameter_source: source,
    count: request.count,
    seed: request.seed,
    draw_count: request.count,
    complete: true,
  };
  return bytes([... [header], ...draws, { trailer }]
    .map((document) => JSON.stringify(document)).join("\n") + "\n");
}

export default [
  {
    name: "fixed and model-prior plans lower to one exact native request",
    fn: async () => {
      const model = bytes('{ "bayeswire_ir": 1 }\n');
      const design = bytes('{"format":"bayescycle.data.json.v1","variables":{}}\n');
      const parameters = bytes('{"format":"bayescycle.data.json.v1","variables":{"theta":{"dtype":"float64","shape":[],"values":[0.5]}}}\n');
      const requests = [];
      const executionSignals = [];
      const executor = {
        execute: async (request, options) => {
          requests.push(request);
          executionSignals.push(options.signal);
          return { rawBytes: generatedOutput(request) };
        },
      };
      const runtime = new BrowserRuntime(executor);
      const plans = [
        generateDatasets(model, { design, parameterSource: fixed(parameters), count: 2, seed: 7 }),
        generateDatasets(model, { design, parameterSource: modelPrior(model), count: 3, seed: 8 }),
      ];
      const artifactNames = [];
      for (const [index, plan] of plans.entries()) {
        const controller = new AbortController();
        const result = await runtime.run(
          { type: "run", id: `generation-${index}`, operation: "generate", plan },
          undefined,
          { signal: controller.signal },
        );
        assert(executionSignals.at(-1) === controller.signal, "generate lost its run signal");
        artifactNames.push(result.artifacts.map((artifact) => artifact.name));
        assert(result.artifacts.some((artifact) => artifact.name === "generated_datasets.ndjson"), "generated artifact missing");
      }
      assert(JSON.stringify(artifactNames[0]) === JSON.stringify([
        "model.ir.json", "design.json", "generation-plan.json", "fixed-parameters.json",
        "generated_datasets.ndjson", "run.json",
      ]), `fixed publication is incomplete: ${JSON.stringify(artifactNames[0])}`);
      assert(JSON.stringify(artifactNames[1]) === JSON.stringify([
        "model.ir.json", "design.json", "generation-plan.json",
        "generated_datasets.ndjson", "run.json",
      ]), `model-prior publication is incomplete: ${JSON.stringify(artifactNames[1])}`);
      assert(requests.length === 2, `expected one request per plan, got ${requests.length}`);
      for (const request of requests) {
        assert(request.command === "generate", `unexpected command: ${request.command}`);
        assert(request.model === new TextDecoder().decode(model), "model bytes changed during lowering");
        assert(request.design === new TextDecoder().decode(design), "design bytes changed during lowering");
        assert(request.identities.generation_model_hash === await hash(model), "model identity changed");
        assert(request.identities.design_hash === await hash(design), "design identity changed");
      }
      assert(requests[0].parameter_source.parameters === new TextDecoder().decode(parameters), "fixed bytes changed");
      assert(requests[0].identities.parameters_hash === await hash(parameters), "fixed identity changed");
      assert(requests[1].parameter_source.kind === "model-prior", "model-prior source changed");
      assert(requests[1].parameter_source.authored_provenance === null, "model-prior provenance changed");
    },
  },
  {
    name: "runtime rejects caller asserted posterior association",
    fn: async () => {
      const model = bytes('{"bayeswire_ir":1,"model":{}}\n');
      const design = bytes('{"format":"bayescycle.data.json.v1","variables":{}}\n');
      const fitData = bytes('{"format":"bayescycle.data.json.v1","variables":{}}\n');
      let requests = 0;
      const runtime = new BrowserRuntime({
        execute: async (request) => {
          requests += 1;
          return { rawBytes: generatedOutput(request) };
        },
      });
      let message = "";
      try {
        await runtime.run({
          type: "run",
          id: "forged-runtime-fit",
          operation: "generate",
          plan: generateDatasets(model, {
            design,
            parameterSource: posteriorOf(
              fitArtifact(model, fitData, runtimePosteriorBytes(), "runtime"),
            ),
            count: 1,
            seed: 0,
          }),
        });
      } catch (error) {
        message = String(error);
      }
      assert(message.includes("association"), `forged runtime fit succeeded: ${message}`);
      assert(requests === 0, `forged runtime fit reached engine ${requests} times`);
    },
  },
  {
    name: "runtime rejects malformed successful generation output",
    fn: async () => {
      const model = bytes('{"bayeswire_ir":1,"model":{}}\n');
      const design = bytes('{"format":"bayescycle.data.json.v1","variables":{}}\n');
      const parameters = bytes(
        '{"format":"bayescycle.data.json.v1","variables":' +
        '{"theta":{"dtype":"float64","shape":[],"values":[0.5]}}}\n',
      );
      const runtime = new BrowserRuntime({
        execute: async () => ({ rawBytes: bytes('{"bad":true}\n') }),
      });
      let message = "";
      try {
        await runtime.run({
          type: "run",
          id: "malformed-generation",
          operation: "generate",
          plan: generateDatasets(model, {
            design, parameterSource: fixed(parameters), count: 1, seed: 0,
          }),
        });
      } catch (error) {
        message = String(error);
      }
      assert(message.includes("generated-dataset"), `malformed output succeeded: ${message}`);
    },
  },
  {
    name: "runtime generates paired datasets through the vendored wasm",
    fn: async () => {
      const model = bytes(await (await fetch("/tests/fixtures/corpus/linear_regression.json")).text());
      const design = bytes(JSON.stringify({
        format: "bayescycle.data.json.v1",
        variables: { x: { dtype: "float64", shape: [3], values: [-1, 0, 1] } },
      }));
      const parameters = bytes(JSON.stringify({
        format: "bayescycle.data.json.v1",
        variables: {
          alpha: { dtype: "float64", shape: [], values: [0.5] },
          beta: { dtype: "float64", shape: [], values: [1.2] },
          sigma: { dtype: "float64", shape: [], values: [0.4] },
        },
      }));
      const result = await new BrowserRuntime().run({
        type: "run",
        id: "native-generation",
        operation: "generate",
        plan: generateDatasets(model, {
          design,
          parameterSource: fixed(parameters),
          count: 1,
          seed: 17,
        }),
      });
      const artifact = result.artifacts.find((entry) => entry.name === "generated_datasets.ndjson");
      assert(artifact !== undefined, "paired generated artifact missing");
      const documents = UTF8.decode(artifact.bytes).trimEnd().split("\n").map((line) => JSON.parse(line));
      assert(documents.length === 3, `unexpected generated artifact length: ${documents.length}`);
      assert(documents[1].dataset.variables.x !== undefined, "complete dataset lost design");
      assert(documents[1].dataset.variables.y !== undefined, "complete dataset lost outcome");
      const modelArtifact = result.artifacts.find((entry) => entry.name === "model.ir.json");
      assert(modelArtifact !== undefined, "published model artifact missing");
      const firstByte = modelArtifact.bytes[0];
      modelArtifact.bytes[0] = 0;
      assert(modelArtifact.bytes[0] === firstByte, "published artifact bytes escaped by alias");
    },
  },
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
    name: "runtime propagates abort signals to diagnostic and recovery verbs",
    fn: async () => {
      const calls = [];
      const executor = {
        execute: async (request, options) => {
          calls.push({ request, options });
          return { rawBytes: bytes("{}") };
        },
      };
      const runtime = new BrowserRuntime(executor);
      const controller = new AbortController();
      await runtime.run(
        { operation: "diagnose", fit: "fit" },
        undefined,
        { signal: controller.signal },
      );
      await runtime.run(
        {
          operation: "recover-check",
          fit: "fit",
          truth: { format: "bayescycle.data.json.v1", variables: {} },
        },
        undefined,
        { signal: controller.signal },
      );
      assert(calls.length === 2, `expected two verb calls, got ${calls.length}`);
      assert(calls.every((call) => call.options.signal === controller.signal), "a follow-up verb lost its run signal");
    },
  },
  {
    name: "retired legacy operations are rejected",
    fn: async () => {
      const runtime = new BrowserRuntime();
      for (const operation of ["prior-predictive", "posterior-predictive", "simulate"]) {
        let rejected = false;
        try {
          await runtime.run({ operation });
        } catch (error) {
          rejected = error.kind === "UnsupportedOperation";
        }
        assert(rejected, `${operation} was not rejected as unsupported`);
      }
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
    name: "selected canonical data bytes remain byte-exact through conditioning",
    fn: async () => {
      const data = bytes(
        '{"format":"bayescycle.data.json.v1","variables":' +
        '{"x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]}}}\n',
      );
      const executor = {
        execute: async () => ({ rawBytes: bytes('{"kind":"header"}\n{"x":1}\n{"kind":"trailer"}\n') }),
      };
      const result = await new BrowserRuntime(executor).run({
        type: "run", id: "exact-data", operation: "sample",
        modelIr: bytes('{"bayeswire_ir":1}'), data,
        settings: { chains: 1, num_warmup: 0, num_draws: 4 },
      });
      const artifact = result.artifacts.find((entry) => entry.name === "data.json");
      assert(artifact !== undefined, "data artifact missing");
      assert(UTF8.decode(artifact.bytes) === UTF8.decode(data), "canonical data lexemes changed");
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
    name: "runtime conditions on one dataset through the workflow operation",
    fn: async () => {
      const runtime = new BrowserRuntime();
      const progress = [];
      const result = await runtime.run({
        type: "run",
        id: "condition-1",
        operation: "condition",
        modelIr: await text("model.ir.json"),
        data: await text("data.json"),
        settings: {
          chains: 1, num_warmup: 4, num_draws: 4, seed: 17,
          max_treedepth: 4, target_accept: 0.8,
        },
      }, (event) => progress.push(event));
      assert(result.artifacts.some((artifact) => artifact.name === "posterior.ndjson"),
        "condition posterior artifact missing");
      assert(result.artifacts.some((artifact) => artifact.name === "diagnostics.json"),
        "condition diagnostics artifact missing");
      assert(result.fitArtifact !== undefined, "condition did not issue runtime fit authority");
      assert(progress.some((event) => event.chainId === 0), "condition progress missing");
    },
  },
  {
    name: "first chain failure aborts every sibling execution",
    fn: async () => {
      const calls = [];
      const executor = {
        execute: (request, options) => new Promise((resolve, reject) => {
          const call = { request, options, aborted: false };
          calls.push(call);
          options.signal.addEventListener("abort", () => {
            call.aborted = true;
            reject(new EngineError("Cancelled", "cancelled sibling"));
          }, { once: true });
          if (request.chain_id === 0) {
            queueMicrotask(() => reject(new EngineError("ChainFailure", "chain zero failed")));
          }
        }),
      };
      let message = "";
      try {
        await new BrowserRuntime(executor).run({
          operation: "sample",
          modelIr: bytes('{"bayeswire_ir":1}'),
          data: bytes('{"format":"bayescycle.data.json.v1","variables":{}}\n'),
          settings: { chains: 3, num_warmup: 0, num_draws: 4 },
        });
      } catch (error) {
        message = String(error);
      }
      assert(message.includes("chain zero failed"), `first failure disappeared: ${message}`);
      assert(calls.length === 3, `only ${calls.length} chains launched`);
      assert(calls.every((call) => call.aborted), "a sibling chain was not aborted");
    },
  },
  {
    name: "run abort propagates to every in-flight chain",
    fn: async () => {
      const calls = [];
      const executor = {
        execute: (request, options) => new Promise((resolve, reject) => {
          const call = { request, options, aborted: false };
          calls.push(call);
          options.signal.addEventListener("abort", () => {
            call.aborted = true;
            reject(new EngineError("Cancelled", "cancelled chain"));
          }, { once: true });
        }),
      };
      const controller = new AbortController();
      const pending = new BrowserRuntime(executor).run({
        operation: "sample",
        modelIr: bytes('{"bayeswire_ir":1}'),
        data: bytes('{"format":"bayescycle.data.json.v1","variables":{}}\n'),
        settings: { chains: 3, num_warmup: 0, num_draws: 4 },
      }, undefined, { signal: controller.signal });
      controller.abort();
      let message = "";
      try { await pending; } catch (error) { message = String(error); }
      assert(message.includes("cancelled chain"), `abort disappeared: ${message}`);
      assert(calls.length === 3, `only ${calls.length} chains launched`);
      assert(calls.every((call) => call.aborted), "an aborted run retained a chain");
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
