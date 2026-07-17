import {
  MAX_GENERATION_COUNT,
  MAX_GENERATION_INPUT_BYTES,
  draw,
  fitArtifact,
  fixed,
  generateDatasets,
  generationInvalidationKey,
  generationPlanIdentity,
  jointPredict,
  modelPrior,
  outcomesOf,
  parseGenerationPlanDocument,
  posteriorOf,
  serializeGenerationPlan,
  validateGenerationPlan,
} from "/site/src/generation/plan.mjs";

const UTF8 = new TextEncoder();
const TEXT = new TextDecoder();
const MODEL_BYTES = UTF8.encode('{"bayeswire_ir":1,"model":{}}\n');
const OTHER_MODEL_BYTES = UTF8.encode('{"bayeswire_ir":1,"model":{"name":"other"}}\n');
const DESIGN_BYTES = UTF8.encode('{"format":"bayescycle.data.json.v1","variables":{"x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]}}}\n');
const FIXED_BYTES = UTF8.encode('{"format":"bayescycle.data.json.v1","variables":{"alpha":{"dtype":"float64","shape":[],"values":[0.5]}}}\n');
const FIT_DATA_BYTES = UTF8.encode('{"format":"bayescycle.data.json.v1","variables":{}}\n');
const POSTERIOR_BYTES = UTF8.encode('{"draws_format":"v0-provisional"}\n');
const IDENTITY = "sha256:609db688556c5d8c2f5460b0029c7899e307b14c1aa54fdcf45bac4a38543697";
const INVALIDATION_KEY = "sha256:ef642308cab544f3e479d74742988477c38a6f3640cb4777365bb8ff5e433377";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function hashBytes(value) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", value));
  return `sha256:${[...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

async function fixtureBytes() {
  const response = await fetch("/tests/fixtures/generation_plan.fixed.v0.json");
  return new Uint8Array(await response.arrayBuffer());
}

function fixedPlan() {
  return draw(
    jointPredict(fixed(FIXED_BYTES), outcomesOf(MODEL_BYTES, DESIGN_BYTES)),
    2,
    7,
  );
}

async function rejects(operation, expected) {
  let message = "";
  try { await operation(); } catch (error) { message = String(error); }
  assert(message.includes(expected), `expected ${expected} error, got ${message}`);
}

export default [
  {
    name: "generation convenience equals draw of joint predict for every source",
    fn: async () => {
      const fit = fitArtifact(MODEL_BYTES, FIT_DATA_BYTES, POSTERIOR_BYTES, "runtime");
      for (const source of [fixed(FIXED_BYTES), modelPrior(MODEL_BYTES), posteriorOf(fit)]) {
        const expected = draw(jointPredict(source, outcomesOf(MODEL_BYTES, DESIGN_BYTES)), 2, 7);
        const actual = generateDatasets(MODEL_BYTES, { design: DESIGN_BYTES, parameterSource: source, count: 2, seed: 7 });
        assert(TEXT.decode(await serializeGenerationPlan(actual)) === TEXT.decode(await serializeGenerationPlan(expected)), "convenience plan differs");
      }
    },
  },
  {
    name: "serialized fixed plan matches shared versioned fixture and identities",
    fn: async () => {
      const plan = fixedPlan();
      const fixture = await fixtureBytes();
      assert(TEXT.decode(await serializeGenerationPlan(plan)) === TEXT.decode(fixture), "shared plan bytes differ");
      const document = await parseGenerationPlanDocument(fixture);
      assert(TEXT.decode(document.bytes) === TEXT.decode(fixture), "parsed plan bytes changed");
      assert(document.identityHash === IDENTITY && document.invalidationKey === INVALIDATION_KEY, "parsed identities differ");
      assert(await generationPlanIdentity(plan) === IDENTITY, "plan identity differs");
      assert(await generationInvalidationKey(plan) === INVALIDATION_KEY, "invalidation key differs");
    },
  },
  {
    name: "rejects designSource as a generation option",
    fn: async () => {
      await rejects(() => generateDatasets(MODEL_BYTES, {
        design: DESIGN_BYTES,
        designSource: { x: "linspace(-2, 2, 3)" },
        parameterSource: fixed(FIXED_BYTES),
        count: 2,
        seed: 7,
      }), "unknown or missing fields");
    },
  },
  {
    name: "serializes exact model-prior and posterior source variants",
    fn: async () => {
      const provenance = {
        claimedSourceModelHash: `sha256:${"1".repeat(64)}`,
        claimedOutcomeModelHash: `sha256:${"2".repeat(64)}`,
      };
      const fit = fitArtifact(MODEL_BYTES, FIT_DATA_BYTES, POSTERIOR_BYTES, "portable");
      const plans = [modelPrior(MODEL_BYTES, provenance), posteriorOf(fit)].map((source) =>
        generateDatasets(MODEL_BYTES, { design: DESIGN_BYTES, parameterSource: source, count: 3, seed: 11 }));
      const documents = [];
      for (const plan of plans) documents.push(JSON.parse(TEXT.decode(await serializeGenerationPlan(plan))));
      assert(JSON.stringify(documents[0].distribution.parameters) === JSON.stringify({
        kind: "model-prior",
        model_hash: "sha256:2c8947663a1a49b8e48c52542efc5b94c237365e40d859ee088b9d214845ccd5",
        authored_provenance: {
          claimed_source_model_hash: `sha256:${"1".repeat(64)}`,
          claimed_outcome_model_hash: `sha256:${"2".repeat(64)}`,
        },
      }), "model-prior descriptor differs");
      assert(JSON.stringify(documents[1].distribution.parameters) === JSON.stringify({
        kind: "posterior",
        fit_hash: "sha256:9cd8922c37ec4ace35caf850100f4230988d0214b1d7dfee1c6015dd7cd48bee",
        fit_model_hash: "sha256:2c8947663a1a49b8e48c52542efc5b94c237365e40d859ee088b9d214845ccd5",
        fit_data_hash: "sha256:7657f9e3dcc7ce5eba549ba1641bd0bf3d7b5fc1dceea7b042184bfbc9c63294",
      }), "posterior descriptor differs");
    },
  },
  {
    name: "composed-prior provenance claims composed source and original outcome hashes",
    fn: async () => {
      const composedHash = await hashBytes(OTHER_MODEL_BYTES);
      const originalHash = await hashBytes(MODEL_BYTES);
      const plan = generateDatasets(OTHER_MODEL_BYTES, {
        design: DESIGN_BYTES,
        parameterSource: modelPrior(OTHER_MODEL_BYTES, {
          claimedSourceModelHash: composedHash,
          claimedOutcomeModelHash: originalHash,
        }),
        count: 2,
        seed: 3,
      });
      const document = JSON.parse(TEXT.decode(await serializeGenerationPlan(plan)));
      assert(
        document.distribution.parameters.model_hash === composedHash,
        "composed prior model hash changed",
      );
      assert(
        document.distribution.outcomes.model_hash === composedHash,
        "generation outcomes did not use composed bytes",
      );
      assert(JSON.stringify(document.distribution.parameters.authored_provenance) === JSON.stringify({
        claimed_source_model_hash: composedHash,
        claimed_outcome_model_hash: originalHash,
      }), "composed-prior authored provenance was wired backwards");
    },
  },
  {
    name: "plans own bytes and reject functions DOM and backend-like values",
    fn: async () => {
      const mutable = Uint8Array.from(FIXED_BYTES);
      const source = fixed(mutable);
      mutable[0] = 0;
      const exposed = source.parametersBytes;
      exposed[0] = 0;
      assert(TEXT.decode(source.parametersBytes) === TEXT.decode(FIXED_BYTES), "fixed bytes escaped by alias");
      for (const value of [() => {}, document.body, { execute() {} }]) {
        await rejects(() => fixed(value), "bytes");
      }
      await rejects(
        () => jointPredict(modelPrior(OTHER_MODEL_BYTES), outcomesOf(MODEL_BYTES, DESIGN_BYTES)),
        "model",
      );
    },
  },
  {
    name: "rejects plan bounds unknown fields and mutable executable shapes",
    fn: async () => {
      const distribution = jointPredict(fixed(FIXED_BYTES), outcomesOf(MODEL_BYTES, DESIGN_BYTES));
      for (const [count, seed, expected] of [[0, 0, "count"], [1001, 0, "count"], [1, -1, "seed"], [1, 2 ** 53, "seed"]]) {
        await rejects(() => draw(distribution, count, seed), expected);
      }
      const executable = fixedPlan();
      await rejects(() => validateGenerationPlan({ ...executable, worker: {} }), "unknown");
      const serialized = JSON.parse(TEXT.decode(await fixtureBytes()));
      delete serialized.generation_plan_format;
      await rejects(
        () => parseGenerationPlanDocument(UTF8.encode(`${JSON.stringify(serialized)}\n`)),
        "format",
      );
      const nested = "[".repeat(65) + "0" + "]".repeat(65);
      const deep = TEXT.decode(await fixtureBytes()).replace(
        '{"kind":"fixed","parameters_hash":',
        `{"kind":"fixed","parameters_hash":${nested},"ignored":`,
      );
      await rejects(() => parseGenerationPlanDocument(UTF8.encode(deep)), "depth");
      const fixture = TEXT.decode(await fixtureBytes());
      await rejects(() => parseGenerationPlanDocument(UTF8.encode(
        fixture.replace('"kind":"draw"', '"kind":"wrong","kind":"draw"'),
      )), "duplicate");
      for (const fractional of [
        fixture.replace('"count":2', '"count":2.0000000000000001'),
        fixture.replace('"seed":7', '"seed":9007199254740991.1'),
      ]) {
        await rejects(() => parseGenerationPlanDocument(UTF8.encode(fractional)), "integer");
      }
      assert(MAX_GENERATION_COUNT === 1000, "count bound changed");
      assert(MAX_GENERATION_INPUT_BYTES === 8 * 1024 * 1024, "input bound changed");
    },
  },
];
