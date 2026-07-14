import {
  MAX_GENERATED_ARTIFACT_BYTES,
  MAX_GENERATED_LINE_BYTES,
  parseGeneratedDatasets,
  verifyGeneratedDatasets,
} from "/site/src/generation/artifact.mjs";

const UTF8 = new TextEncoder();
const TEXT = new TextDecoder();

const MODEL_BYTES = UTF8.encode('{"bayeswire_ir":1,"model":{}}\n');
const DESIGN_BYTES = UTF8.encode('{"format":"bayescycle.data.json.v1","variables":{"x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]}}}\n');
const FIXED_BYTES = UTF8.encode('{"format":"bayescycle.data.json.v1","variables":{"alpha":{"dtype":"float64","shape":[],"values":[0.5]}}}\n');
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
      ];
      const unknown = documents(fixture);
      unknown[0].unexpected = true;
      malformed.push([encodeDocuments(unknown), "unknown"]);
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
