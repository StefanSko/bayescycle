import {
  MAX_GENERATED_ARTIFACT_BYTES,
  MAX_GENERATED_LINE_BYTES,
  parseGeneratedDatasets,
  verifyGeneratedDatasets,
} from "/site/src/generation/artifact.mjs";
import { validatePortablePosterior } from "/site/src/generation/posterior-source.mjs";

const UTF8 = new TextEncoder();
const TEXT = new TextDecoder();

const MODEL_BYTES = UTF8.encode('{"bayeswire_ir":1,"model":{}}\n');
const DESIGN_BYTES = UTF8.encode('{"format":"bayescycle.data.json.v1","variables":{"x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]}}}\n');
const FIXED_BYTES = UTF8.encode('{"format":"bayescycle.data.json.v1","variables":{"alpha":{"dtype":"float64","shape":[],"values":[0.5]}}}\n');
const FIT_DATA_BYTES = UTF8.encode('{"format":"bayescycle.data.json.v1","variables":{}}\n');
const PARAMETERS_0 = TEXT.decode(FIXED_BYTES);
const DATASET_0 = '{"format":"bayescycle.data.json.v1","variables":{"x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]},"y":{"dtype":"float64","shape":[3],"values":[-0.2,0.5,1.2]}}}\n';

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function fixtureBytes() {
  const response = await fetch("/tests/fixtures/generated_datasets.v0.ndjson");
  return new Uint8Array(await response.arrayBuffer());
}

function documents(bytes) {
  return TEXT.decode(bytes).trimEnd().split("\n").map((line) => JSON.parse(line));
}

function encodeDocuments(values) {
  return UTF8.encode(values.map((value) => JSON.stringify(value)).join("\n") + "\n");
}

function sourceStream(bytes, kind) {
  const values = documents(bytes);
  const header = values[0];
  const trailer = values.at(-1).trailer;
  let source;
  let lineages;
  if (kind === "model-prior") {
    source = { kind, model_hash: header.generation_model_hash, authored_provenance: null };
    lineages = [
      { kind, source_draw_index: 0 },
      { kind, source_draw_index: 1 },
    ];
  } else {
    source = {
      kind,
      fit_hash: `sha256:${"1".repeat(64)}`,
      fit_model_hash: header.generation_model_hash,
      fit_data_hash: `sha256:${"2".repeat(64)}`,
    };
    lineages = [
      { kind, source_draw_index: 7, chain: 1, draw: 3 },
      { kind, source_draw_index: 2, chain: 0, draw: 2 },
    ];
  }
  header.parameter_source = source;
  trailer.parameter_source = source;
  for (let index = 0; index < 2; index += 1) {
    values[index + 1].source_lineage = lineages[index];
    values[index + 1].parameters.variables.alpha.values = [index + 1];
  }
  return encodeDocuments(values);
}

async function posteriorSource() {
  const prefix = UTF8.encode("bayescycle-model-data-v1\n");
  const framed = new Uint8Array(
    prefix.length + MODEL_BYTES.length + 1 + FIT_DATA_BYTES.length,
  );
  framed.set(prefix);
  framed.set(MODEL_BYTES, prefix.length);
  framed[prefix.length + MODEL_BYTES.length] = 0x0a;
  framed.set(FIT_DATA_BYTES, prefix.length + MODEL_BYTES.length + 1);
  const fingerprint = await crypto.subtle.digest("SHA-256", framed);
  const digest = `sha256:${[...new Uint8Array(fingerprint)]
    .map((value) => value.toString(16).padStart(2, "0")).join("")}`;
  return encodeDocuments([
    {
      draws_format: "v0-provisional",
      artifact_kind: "posterior_draws",
      artifact_scope: "observed_data_conditioned_parameter_draws",
      model_data_fingerprint: digest,
      params: [{ name: "alpha", shape: [], coordinate_order: [[]] }],
      parameter_count: 1,
      parameter_order: ["alpha"],
      settings: { num_warmup: 0, num_draws: 2, max_treedepth: 4 },
      seed: 0,
      chain_count: 1,
      chain_order: [0],
      draw_count: 2,
    },
    {
      draws_format: "v0-provisional",
      artifact_kind: "posterior_draws",
      artifact_scope: "observed_data_conditioned_parameter_draws",
      draw_index: 0, chain: 0, draw: 0, parameter_count: 1,
      parameter_order: ["alpha"], values: { alpha: 1.0 },
    },
    {
      draws_format: "v0-provisional",
      artifact_kind: "posterior_draws",
      artifact_scope: "observed_data_conditioned_parameter_draws",
      draw_index: 1, chain: 0, draw: 1, parameter_count: 1,
      parameter_order: ["alpha"], values: { alpha: 2.0 },
    },
    {
      trailer: {
        draws_format: "v0-provisional",
        artifact_kind: "posterior_draws",
        artifact_scope: "observed_data_conditioned_parameter_draws",
        model_data_fingerprint: digest,
        seed: 0,
        draws_per_chain: 2,
        chain_count: 1,
        chain_order: [0],
        draw_count: 2,
        parameter_count: 1,
        parameter_order: ["alpha"],
        params: 1,
        chains: [{
          chain: 0, draw_count: 2, divergences: 0, treedepth_histogram: [0, 2],
        }],
      },
    },
  ]);
}

async function posteriorGenerated(fixture, fit, useSourceValues) {
  const values = documents(fixture);
  const descriptor = {
    kind: "posterior",
    fit_hash: await sha256(fit),
    fit_model_hash: await sha256(MODEL_BYTES),
    fit_data_hash: await sha256(FIT_DATA_BYTES),
  };
  values[0].parameter_source = descriptor;
  values.at(-1).trailer.parameter_source = descriptor;
  for (let index = 0; index < 2; index += 1) {
    values[index + 1].source_lineage = {
      kind: "posterior", source_draw_index: index, chain: 0, draw: index,
    };
    if (useSourceValues) {
      values[index + 1].parameters.variables.alpha.values = [index + 1];
    }
  }
  return encodeDocuments(values);
}

async function sha256(value) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", value));
  return `sha256:${[...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

async function rejects(operation, expected) {
  let message = "";
  try { await operation(); } catch (error) { message = String(error); }
  assert(message.includes(expected), `expected ${expected} error, got ${message}`);
}

export default [
  {
    name: "parses canonical generated header pairs and trailer",
    fn: async () => {
      const artifact = parseGeneratedDatasets(await fixtureBytes());
      assert(artifact.generationModelHash === "sha256:2c8947663a1a49b8e48c52542efc5b94c237365e40d859ee088b9d214845ccd5", "model hash changed");
      assert(artifact.designHash === "sha256:02b4040e221b6967267a7e8bcb1692c4b00c9d76e90ed58a7344e5390eaddcc6", "design hash changed");
      assert(artifact.sourceKind === "fixed" && artifact.count === 2 && artifact.seed === 7, "header identity changed");
      assert(artifact.draws.map((draw) => draw.drawIndex).join(",") === "0,1", "draw order changed");
      assert(artifact.draws.every((draw) => draw.parameters.variables.alpha.values[0] === 0.5), "fixed values changed");
      assert(artifact.draws[0].dataset.variables.y.values.join(",") === "-0.2,0.5,1.2", "dataset changed");
    },
  },
  {
    name: "accepts varying model-prior and posterior parameters",
    fn: async () => {
      const fixture = await fixtureBytes();
      for (const kind of ["model-prior", "posterior"]) {
        const artifact = parseGeneratedDatasets(sourceStream(fixture, kind));
        assert(artifact.sourceKind === kind, `${kind} source changed`);
        assert(artifact.draws.map((draw) => draw.parameters.variables.alpha.values[0]).join(",") === "1,2", `${kind} parameters did not vary`);
        assert(artifact.draws.every((draw) => draw.sourceKind === kind), `${kind} lineage changed`);
      }
    },
  },
  {
    name: "selects exact nested bytes and preserves the download",
    fn: async () => {
      const input = await fixtureBytes();
      const original = Uint8Array.from(input);
      const artifact = parseGeneratedDatasets(input);
      input[0] = 0;
      const selected = artifact.select(0);
      assert(TEXT.decode(selected.parametersBytes) === PARAMETERS_0, "parameter bytes changed");
      assert(TEXT.decode(selected.datasetBytes) === DATASET_0, "dataset bytes changed");
      assert(TEXT.decode(artifact.bytes) === TEXT.decode(original), "download bytes changed through aliasing");
      await rejects(() => artifact.select(2), "range");
    },
  },
  {
    name: "verifies hashes fixed values and design prefix",
    fn: async () => {
      const artifact = parseGeneratedDatasets(await fixtureBytes());
      await verifyGeneratedDatasets(artifact, {
        modelBytes: MODEL_BYTES,
        designBytes: DESIGN_BYTES,
        fixedParametersBytes: FIXED_BYTES,
      });
      await rejects(() => verifyGeneratedDatasets(artifact, {
        modelBytes: MODEL_BYTES,
        designBytes: UTF8.encode("{}\n"),
        fixedParametersBytes: FIXED_BYTES,
      }), "design hash");
      await rejects(() => verifyGeneratedDatasets(artifact, {
        modelBytes: MODEL_BYTES,
        designBytes: DESIGN_BYTES,
        fixedParametersBytes: UTF8.encode('{"format":"bayescycle.data.json.v1","variables":{}}\n'),
      }), "fixed parameters");
    },
  },
  {
    name: "resolver rejects output from the wrong generation plan",
    fn: async () => {
      const fixture = await fixtureBytes();
      const prior = parseGeneratedDatasets(sourceStream(fixture, "model-prior"));
      await rejects(() => verifyGeneratedDatasets(prior, {
        modelBytes: MODEL_BYTES,
        designBytes: DESIGN_BYTES,
        fixedParametersBytes: FIXED_BYTES,
        expectedSourceKind: "fixed",
        expectedCount: 2,
        expectedSeed: 7,
      }), "source kind");
      await verifyGeneratedDatasets(prior, {
        modelBytes: MODEL_BYTES,
        designBytes: DESIGN_BYTES,
        expectedSourceKind: "model-prior",
        expectedCount: 2,
        expectedSeed: 7,
        modelPriorBytes: MODEL_BYTES,
        authoredProvenance: null,
      });
      await rejects(() => verifyGeneratedDatasets(prior, {
        modelBytes: MODEL_BYTES,
        designBytes: DESIGN_BYTES,
        expectedSourceKind: "model-prior",
        expectedCount: 2,
        expectedSeed: 7,
        modelPriorBytes: UTF8.encode("other model"),
        authoredProvenance: null,
      }), "model-prior");
      const fit = await posteriorSource();
      const posterior = parseGeneratedDatasets(await posteriorGenerated(fixture, fit, true));
      await verifyGeneratedDatasets(posterior, {
        modelBytes: MODEL_BYTES,
        designBytes: DESIGN_BYTES,
        expectedSourceKind: "posterior",
        expectedCount: 2,
        expectedSeed: 7,
        posteriorBytes: fit,
        fitDataBytes: FIT_DATA_BYTES,
      });
      const wrongPosterior = parseGeneratedDatasets(
        await posteriorGenerated(fixture, fit, false),
      );
      await rejects(() => verifyGeneratedDatasets(wrongPosterior, {
        modelBytes: MODEL_BYTES,
        designBytes: DESIGN_BYTES,
        expectedSourceKind: "posterior",
        expectedCount: 2,
        expectedSeed: 7,
        posteriorBytes: fit,
        fitDataBytes: FIT_DATA_BYTES,
      }), "posterior");
      const fixed = parseGeneratedDatasets(fixture);
      await rejects(() => verifyGeneratedDatasets(fixed, {
        modelBytes: MODEL_BYTES,
        designBytes: DESIGN_BYTES,
        fixedParametersBytes: FIXED_BYTES,
        expectedSourceKind: "fixed",
        expectedCount: 1,
        expectedSeed: 8,
      }), "count");
    },
  },
  {
    name: "portable posterior requires complete chain and trailer lineage",
    fn: async () => {
      const root = "/tests/fixtures/engine/eight_schools_non_centered/";
      const [model, data, posterior] = await Promise.all(
        ["model.ir.json", "data.json", "posterior.ndjson"].map(async (name) =>
          new Uint8Array(await (await fetch(`${root}${name}`)).arrayBuffer())),
      );
      await validatePortablePosterior({
        modelBytes: model, dataBytes: data, posteriorBytes: posterior,
      });
      const originals = documents(posterior);
      const renamed = structuredClone(originals);
      renamed[0].params[0].name = "seed";
      renamed[0].parameter_order[0] = "seed";
      for (const draw of renamed.slice(1, -1)) {
        draw.parameter_order[0] = "seed";
        const { mu, ...remaining } = draw.values;
        draw.values = { seed: mu, ...remaining };
      }
      renamed.at(-1).trailer.parameter_order[0] = "seed";
      await validatePortablePosterior({
        modelBytes: model,
        dataBytes: data,
        posteriorBytes: encodeDocuments(renamed),
      });
      const mutations = [
        (value) => { value[0].artifact_kind = "wrong"; },
        (value) => { delete value[0].settings; },
        (value) => { value[0].chain_count = 99; },
        (value) => { value[1].chain = 99; value[1].draw = 42; },
        (value) => { value.at(-1).trailer.parameter_order = ["wrong"]; },
        (value) => { value.at(-1).trailer.posterior_identity_hash = "fnv1a64:wrong"; },
        (value) => { value.at(-1).trailer.chains[0].draw_count = 1; },
        (value) => { value[1].values.z = [value[1].values.z]; },
      ];
      for (const mutate of mutations) {
        const changed = structuredClone(originals);
        mutate(changed);
        await rejects(() => validatePortablePosterior({
          modelBytes: model,
          dataBytes: data,
          posteriorBytes: encodeDocuments(changed),
        }), "posterior");
      }
      const posteriorText = TEXT.decode(posterior);
      for (const [changed, expected] of [
        [posteriorText.replace('"chain_count":2', '"chain_count":2.0000000000000001'), "integer"],
        [posteriorText.replace('"draw_index":0,', ""), "draw_index"],
        [posteriorText.replace('"target_accept":0.8', '"target_accept":1e400'), "finite"],
        [posteriorText.replace('"num_warmup":300', '"num_warmup":300.0'), "integer"],
        [posteriorText.replace('"max_treedepth":10', '"max_treedepth":10.0'), "integer"],
        [posteriorText.replace('"tree_depth":3', '"tree_depth":3.0'), "integer"],
        [posteriorText.replace('"divergences":0', '"divergences":0.0'), "integer"],
        [posteriorText.replace('"chains":2', '"chains":2.0'), "integer"],
        [posteriorText.replace('"seed":20260702', '"seed":20260702.0'), "integer"],
        [posteriorText.replace('"draw_count":600', '"draw_count":600.0'), "integer"],
        [posteriorText.replace('"chain_count":2', '"chain_count":2.0'), "integer"],
        [posteriorText.replace('"chain_order":[0,1]', '"chain_order":[0.0,1]'), "integer"],
        [posteriorText.replace(
          '"model_data_fingerprint":',
          '"model_data_fingerprint":"sha256:' + "0".repeat(64) + '","model_data_fingerprint":',
        ), "duplicate"],
      ]) {
        await rejects(() => validatePortablePosterior({
          modelBytes: model, dataBytes: data, posteriorBytes: UTF8.encode(changed),
        }), expected);
      }
    },
  },
  {
    name: "rejects malformed truncated nonfinite and oversized streams",
    fn: async () => {
      const fixture = await fixtureBytes();
      const text = TEXT.decode(fixture);
      const lines = text.trimEnd().split("\n");
      const malformed = [
        [UTF8.encode(lines.slice(0, -1).join("\n") + "\n"), "trailer"],
        [UTF8.encode(text.replace('"count":2', '"count":3')), "count"],
        [UTF8.encode(text.replace('"draw_index":1', '"draw_index":3')), "draw_index"],
        [UTF8.encode(text.replace('"values":[0.5]', '"values":[NaN]')), "finite"],
        [UTF8.encode(text.replace('"count":2', '"count":2.0000000000000001')), "integer"],
        [UTF8.encode(text.replace('"seed":7', '"seed":9007199254740991.1')), "integer"],
      ];
      const unknown = documents(fixture);
      unknown[0].unexpected = true;
      malformed.push([encodeDocuments(unknown), "unknown"]);
      const roundedInteger = documents(fixture);
      roundedInteger[0].parameter_schema[0].dtype = "int64";
      for (const draw of roundedInteger.slice(1, -1)) {
        draw.parameters.variables.alpha.dtype = "int64";
        draw.parameters.variables.alpha.values = [1];
      }
      malformed.push([
        UTF8.encode(TEXT.decode(encodeDocuments(roundedInteger)).replace(
          '"dtype":"int64","shape":[],"values":[1]',
          '"values":[1.0000000000000001],"dtype":"int64","shape":[]',
        )),
        "integer",
      ]);
      const attacker = '{"format":"bayescycle.data.json.v1","variables":{}}';
      malformed.push([
        UTF8.encode(text.replace('"parameters":', `"parameters":${attacker},"parameters":`)),
        "duplicate",
      ]);
      const oversized = new Uint8Array(MAX_GENERATED_LINE_BYTES + fixture.byteLength);
      oversized.fill(0x20, 0, MAX_GENERATED_LINE_BYTES);
      oversized.set(fixture, MAX_GENERATED_LINE_BYTES);
      malformed.push([oversized, "line"]);
      for (const [stream, expected] of malformed) {
        await rejects(() => parseGeneratedDatasets(stream), expected);
      }
      assert(MAX_GENERATED_ARTIFACT_BYTES === 64 * 1024 * 1024, "artifact bound changed");
    },
  },
];
