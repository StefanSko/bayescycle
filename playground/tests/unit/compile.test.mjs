import { compile } from "/site/src/compile/index.mjs";

const FIXTURE_ROOT = "/tests/fixtures/corpus/";
const UTF8 = new TextDecoder();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function fetchText(path) {
  const response = await fetch(`${FIXTURE_ROOT}${path}`);
  assert(response.ok, `fixture request for ${path} failed with HTTP ${response.status}`);
  return response.text();
}

function canonicalJson(value) {
  if (Array.isArray(value)) return value.map(canonicalJson);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalJson(value[key])]),
    );
  }
  return value;
}

function canonicalString(value) {
  return JSON.stringify(canonicalJson(value));
}

function equalLinearSchema(schema) {
  assert(
    canonicalString(schema.data) === canonicalString([
      { name: "x", dtype: "float64", kind: "vector", length: null },
    ]),
    `unexpected linear data schema: ${JSON.stringify(schema.data)}`,
  );
  assert(
    JSON.stringify(schema.observed) === JSON.stringify([{ name: "y" }]),
    `unexpected linear observed schema: ${JSON.stringify(schema.observed)}`,
  );
  assert(
    schema.parameters.map((parameter) => parameter.name).join(",") === "alpha,beta,sigma" &&
      schema.parameters[0].prior === "Normal(0.0, 1.0)" &&
      schema.parameters[0].default === 0 &&
      schema.parameters[2].constraint === "> 0" &&
      schema.parameters[2].default === null,
    `unexpected linear parameter schema: ${JSON.stringify(schema.parameters)}`,
  );
}

export default [
  {
    name: "all 11 corpus models match native hashes and golden IR",
    fn: async () => {
      const hashes = JSON.parse(await fetchText("hashes.json"));
      const names = Object.keys(hashes);
      assert(names.length === 11, `expected 11 corpus models, found ${names.length}`);

      for (const name of names) {
        const [source, goldenText] = await Promise.all([
          fetchText(`${name}.py`),
          fetchText(`${name}.json`),
        ]);
        const result = await compile(source);
        assert(result.ok, `${name} compilation failed:\n${result.traceback}`);
        assert(
          result.modelSchema.schema_format === "bayescycle.playground.model-schema.v0",
          `${name} did not return the worker-derived model schema`,
        );
        assert(
          result.modelSchema.parameters.every((parameter) =>
            typeof parameter.prior === "string" && Array.isArray(parameter.shape)),
          `${name} returned malformed parameter schema entries`,
        );
        assert(
          result.irHash === hashes[name],
          `${name} hash mismatch: expected ${hashes[name]}, got ${result.irHash}`,
        );
        if (name === "linear_regression") {
          equalLinearSchema(result.modelSchema);
        }
        const actual = canonicalString(JSON.parse(UTF8.decode(result.irBytes)));
        const golden = canonicalString(JSON.parse(goldenText));
        assert(actual === golden, `${name} IR differs from its golden document`);
      }
    },
  },
  {
    name: "surfaces bayeswire declaration errors verbatim",
    fn: async () => {
      const source = `from bayeswire import Param, model\nfrom bayeswire.distributions import Normal\n\nclass Base:\n    pass\n\n@model\nclass Invalid(Base):\n    x = Param(Normal(0.0, 1.0))\n`;
      const result = await compile(source);
      assert(!result.ok, "expected inheritance declaration to fail");
      assert(
        result.exceptionType === "builtins.TypeError",
        `unexpected exception type: ${result.exceptionType}`,
      );
      const expectedMessage =
        "Model declaration classes must not use inheritance: 'Invalid' inherits from 'Base'. " +
        "All declarations must live in the decorated class body; inherited declarations would be silently ignored otherwise";
      assert(
        result.message === expectedMessage,
        `unexpected declaration error: ${result.message}`,
      );
      assert(
        result.traceback.includes("bayeswire/model/decorator.py"),
        `traceback does not identify bayeswire decorator:\n${result.traceback}`,
      );
    },
  },
  {
    name: "requires exactly one model class",
    fn: async () => {
      const result = await compile("answer = 42\n");
      assert(!result.ok, "expected source without a model to fail");
      assert(
        result.exceptionType === "builtins.ValueError",
        `unexpected exception type: ${result.exceptionType}`,
      );
      assert(
        result.message === "Expected exactly one @model class, found 0",
        `unexpected model-count error: ${result.message}`,
      );
    },
  },
  {
    name: "index vector design slots are marked integer",
    fn: async () => {
      const result = await compile(`from bayeswire import Data, Observed, Param, model
from bayeswire.distributions import Binomial, Normal

@model
class IndexedGroups:
    theta = Param(Normal(0.0, 1.0), size=3)
    idx = Data.vector()
    x = Data.vector()
    w = Data.vector(3)
    trials = Data.vector()
    y = Observed(Normal(theta[idx] + x + w, 1.0))
    k = Observed(Binomial(trials, 0.5))
`);
      assert(result.ok, `indexed model compilation failed: ${result.message}`);
      const byName = Object.fromEntries(
        result.modelSchema.data.map((slot) => [slot.name, slot]),
      );
      assert(byName.idx.dtype === "int64", `index slot dtype: ${byName.idx.dtype}`);
      assert(byName.x.dtype === "float64", `value slot dtype: ${byName.x.dtype}`);
      assert(byName.trials.dtype === "int64", `trial-count slot dtype: ${byName.trials.dtype}`);
      assert(byName.w.length === 3, `exact-shape slot length: ${byName.w.length}`);
      assert(byName.x.length === null, `unresolved slot length: ${byName.x.length}`);
    },
  },
  {
    name: "selects the sole unreferenced composed root",
    fn: async () => {
      const result = await compile(`from bayeswire import Param, Submodel, model\nfrom bayeswire.distributions import Normal\n\n@model\nclass Component:\n    x = Param(Normal(0.0, 1.0))\n\n@model\nclass Root:\n    component = Submodel(Component)\n`);
      assert(result.ok, `composed root compilation failed: ${result.message}`);
      const document = JSON.parse(UTF8.decode(result.irBytes));
      assert(JSON.stringify(document).includes("component.x"), "compiled IR did not select the composed root");
    },
  },
  {
    name: "selects a with_prior result over its source and target",
    fn: async () => {
      const result = await compile(`from bayeswire import Observed, Param, model, with_prior
from bayeswire.distributions import Normal

@model
class Target:
    theta = Param(Normal(0.0, 1.0))
    y = Observed(Normal(theta, 1.0))

@model
class Prior:
    theta = Param(Normal(2.0, 0.5))

Composed = with_prior(Target, prior=Prior)
`);
      assert(result.ok, `with_prior root compilation failed: ${result.message}`);
      const document = JSON.parse(UTF8.decode(result.irBytes));
      const serialized = JSON.stringify(document);
      assert(serialized.includes('"value":2'), "compiled IR did not use the source prior");
      assert(serialized.includes('"name":"y"'), "compiled IR did not retain the target outcome");
    },
  },
  {
    name: "rejects a dependency cycle for one local model",
    fn: async () => {
      const result = await compile(`from bayeswire import Param, model
from bayeswire.distributions import Normal

@model
class Only:
    theta = Param(Normal(0.0, 1.0))

type.__setattr__(Only, "_model_dependencies", (Only,))
`);
      assert(!result.ok, "single-model dependency cycle was accepted");
      assert(result.message.includes("model dependency cycle"), `unexpected cycle error: ${result.message}`);
    },
  },
  {
    name: "compiler executes in a dedicated worker",
    fn: async () => {
      const source = await fetchText("linear_regression.py");
      const result = await compile(source);
      assert(result.ok, `worker compilation failed: ${result.traceback}`);
      assert(result.executionContext === "worker", `unexpected context: ${result.executionContext}`);
    },
  },
  {
    name: "module poisoning cannot affect the next compiler worker",
    fn: async () => {
      const source = `import bayeswire.ir\nbayeswire.ir.canonical_bytes = lambda _meta: b"{}"\n${await fetchText("linear_regression.py")}`;
      const poisoned = await compile(source);
      assert(poisoned.ok, `poisoning fixture failed: ${poisoned.message}`);
      const hashes = JSON.parse(await fetchText("hashes.json"));
      const clean = await compile(await fetchText("linear_regression.py"));
      assert(clean.ok && clean.irHash === hashes.linear_regression, "module mutation leaked into the next compiler worker");
    },
  },
  {
    name: "compiler timeout resets the isolated worker",
    fn: async () => {
      let timedOut = false;
      try { await compile("while True:\n    pass\n", { timeoutMs: 50 }); }
      catch (error) { timedOut = String(error).includes("timed out"); }
      assert(timedOut, "non-terminating source did not time out");
      const result = await compile(await fetchText("linear_regression.py"));
      assert(result.ok, `compiler did not recover after timeout: ${result.message}`);
    },
  },
];
