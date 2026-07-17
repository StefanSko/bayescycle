import { compile, compileScenario } from "/site/src/compile/index.mjs";

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
    name: "scenario compile composes one partial prior with the current model",
    fn: async () => {
      const source = `from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Normal, Truncated

@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())
    x = Data.vector()
    y = Observed(Normal(alpha + beta * x, sigma))
`;
      const [target, result] = await Promise.all([
        compile(source),
        compileScenario(source, `@model
class SteepSlopes:
    beta = Param(Normal(3.0, 0.25))
`),
      ]);
      assert(target.ok, `target compilation failed: ${target.traceback}`);
      assert(result.ok, `scenario compilation failed: ${result.traceback}`);
      assert(result.targetIrHash === target.irHash, "scenario target hash changed");
      assert(UTF8.decode(result.targetIrBytes) === UTF8.decode(target.irBytes),
        "scenario target bytes changed");
      const document = JSON.parse(UTF8.decode(result.irBytes));
      const serialized = JSON.stringify(document);
      assert(serialized.includes('"name":"y"'), "composed IR lost the target outcome");
      assert(serialized.includes('"value":3'), "composed IR lost the alternative beta prior");
      assert(
        result.modelSchema.parameters.map((parameter) => parameter.name).join(",") ===
          "alpha,beta,sigma",
        `composed schema changed parameters: ${JSON.stringify(result.modelSchema.parameters)}`,
      );
    },
  },
  {
    name: "scenario compile accepts hierarchical support parameters",
    fn: async () => {
      const source = `from bayeswire import Observed, Param, model
from bayeswire.distributions import Normal
@model
class Target:
    theta = Param(Normal(0.0, 1.0))
    y = Observed(Normal(theta, 1.0))
`;
      const result = await compileScenario(source, `@model
class Hierarchical:
    location = Param(Normal(2.0, 0.5))
    theta = Param(Normal(location, 0.1))
`);
      assert(result.ok, `hierarchical scenario failed: ${result.traceback}`);
      const names = result.modelSchema.parameters.map((parameter) => parameter.name);
      assert(names.indexOf("location") < names.indexOf("theta"),
        `hierarchical parameter order is not ancestral: ${names}`);
    },
  },
  {
    name: "scenario compile rejects source-declared data slots",
    fn: async () => {
      const source = `from bayeswire import Data, Observed, Param, model
from bayeswire.distributions import Normal
@model
class Target:
    beta = Param(Normal(0.0, 1.0))
    x = Data.vector()
    y = Observed(Normal(beta + x, 1.0))
`;
      const result = await compileScenario(source, `@model
class DataPrior:
    prior_location = Data.scalar()
    beta = Param(Normal(prior_location, 0.25))
`);
      assert(!result.ok, "source data reached a composed generation model");
      assert(result.message === "prior-only model must not declare data slots: prior_location",
        `source data error changed: ${result.message}`);
    },
  },
  {
    name: "scenario compile rejects declared parameter dimension mismatches in every target order",
    fn: async () => {
      const target = (declarations) => `from bayeswire import Dim, Param, model
from bayeswire.distributions import Normal
target_axis = Dim("target_axis", coords=("a", "b"))
@model
class Target:
${declarations}
`;
      const snippets = [
        `source_axis = Dim("source_axis", coords=("a", "b"))
@model
class WrongDimensionName:
    beta = Param(Normal(2.0, 0.5), size=2, dims=(source_axis,))
`,
        `target_axis = Dim("target_axis", coords=("a", "different"))
@model
class WrongCoordinates:
    beta = Param(Normal(2.0, 0.5), size=2, dims=(target_axis,))
`,
      ];
      for (const declarations of [
        "    alpha = Param(Normal(0.0, 1.0), size=2, dims=(target_axis,))\n    beta = Param(Normal(0.0, 1.0), size=2, dims=(target_axis,))",
        "    beta = Param(Normal(0.0, 1.0), size=2, dims=(target_axis,))\n    alpha = Param(Normal(0.0, 1.0), size=2, dims=(target_axis,))",
      ]) {
        for (const snippet of snippets) {
          const result = await compileScenario(target(declarations), snippet);
          assert(!result.ok, `dimension mismatch succeeded for target order: ${declarations}`);
          assert(
            result.message === "prior parameter 'beta' must match the target dimensions exactly",
            `dimension error changed: ${result.message}`,
          );
        }
      }
    },
  },
  {
    name: "scenario compile attributes shared-coordinate conflicts to authored support parameters",
    fn: async () => {
      const target = (declarations) => `from bayeswire import Dim, Param, model
from bayeswire.distributions import Normal
shared = Dim("shared", coords=("a", "b"))
@model
class Target:
${declarations}
`;
      const snippet = `shared = Dim("shared", coords=("x", "y"))
@model
class HierarchicalPrior:
    location = Param(Normal(2.0, 0.5), size=2, dims=(shared,))
    beta = Param(Normal(location[0], 0.1))
`;
      for (const declarations of [
        "    alpha = Param(Normal(0.0, 1.0), size=2, dims=(shared,))\n    beta = Param(Normal(0.0, 1.0))",
        "    beta = Param(Normal(0.0, 1.0))\n    alpha = Param(Normal(0.0, 1.0), size=2, dims=(shared,))",
      ]) {
        const result = await compileScenario(target(declarations), snippet);
        assert(!result.ok, `support coordinate conflict succeeded: ${declarations}`);
        assert(
          result.message ===
            "prior parameter 'location' dimension 'shared' conflicts with target coordinates",
          `support coordinate error changed: ${result.message}`,
        );
      }
    },
  },
  {
    name: "scenario compile rejects unused prior parameters on the complete path",
    fn: async () => {
      const source = `from bayeswire import Param, model
from bayeswire.distributions import Normal
@model
class Target:
    theta = Param(Normal(0.0, 1.0))
`;
      const typo = await compileScenario(
        source,
        "@model\nclass Typo:\n    theta = Param(Normal(1.0, 1.0))\n    theeta = Param(Normal(2.0, 1.0))\n",
      );
      assert(!typo.ok, "unused prior parameter was accepted on the complete path");
      assert(
        typo.message.includes("theeta"),
        `unused-parameter error does not name the typo: ${typo.message}`,
      );
      const hierarchical = await compileScenario(
        source,
        "@model\nclass Hierarchical:\n    location = Param(Normal(3.0, 1.0))\n    theta = Param(Normal(location, 0.5))\n",
      );
      assert(
        hierarchical.ok,
        `supporting hierarchical parameter was rejected: ${hierarchical.message}`,
      );
    },
  },
  {
    name: "scenario compile reports prior-only model counts and unknown names",
    fn: async () => {
      const source = `from bayeswire import Param, model
from bayeswire.distributions import Normal
@model
class Target:
    theta = Param(Normal(0.0, 1.0))
`;
      for (const [snippet, count] of [
        ["answer = 42\n", 0],
        ["@model\nclass First:\n    theta = Param(Normal(1.0, 1.0))\n\n@model\nclass Second:\n    theta = Param(Normal(2.0, 1.0))\n", 2],
      ]) {
        const result = await compileScenario(source, snippet);
        assert(!result.ok, `prior-only source with ${count} models succeeded`);
        assert(
          result.message === `Expected exactly one @model class in prior-only source, found ${count}`,
          `unexpected model-count error: ${result.message}`,
        );
      }
      const unknown = await compileScenario(source, `@model
class Typo:
    theeta = Param(Normal(2.0, 1.0))
`);
      assert(!unknown.ok, "unknown prior parameter succeeded");
      assert(
        unknown.message.includes("theeta") && unknown.message.includes("does not exist"),
        `unknown prior parameter error was unclear: ${unknown.message}`,
      );
    },
  },
  {
    name: "scenario compile retains target outcome dimensions",
    fn: async () => {
      const source = `from bayeswire import Dim, Observed, Param, model
from bayeswire.distributions import Normal
observation = Dim("observation")
@model
class DimensionedTarget:
    theta = Param(Normal(0.0, 1.0))
    y = Observed(Normal(theta, 1.0), dims=(observation,))
`;
      const result = await compileScenario(source, `@model
class Shifted:
    theta = Param(Normal(2.0, 0.5))
`);
      assert(result.ok, `dimensioned target composition failed: ${result.traceback}`);
      assert(result.modelSchema.observed.some((entry) => entry.name === "y"),
        "composed schema lost the dimensioned target outcome");
    },
  },
  {
    name: "scenario compile rejects forbidden prior-only factors verbatim",
    fn: async () => {
      const source = `from bayeswire import Param, model
from bayeswire.distributions import Normal
@model
class Target:
    theta = Param(Normal(0.0, 1.0))
`;
      const result = await compileScenario(source, `from bayeswire import Data, PartiallyObserved
@model
class InvalidPrior:
    theta = Param(Normal(2.0, 0.5))
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed = Data.vector(n_obs)
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    value = PartiallyObserved.vector(
        Normal(0.0, 1.0), length=n, observed=observed,
        observed_idx=observed_idx, missing_idx=missing_idx,
    )
`);
      assert(!result.ok, "forbidden prior PartiallyObserved value was stripped");
      assert(result.message.includes("prior") && result.message.includes("PartiallyObserved"),
        `Bayeswire source-prior error was not preserved: ${result.message}`);
    },
  },
  {
    name: "scenario compile retains target non-parameter free values",
    fn: async () => {
      const source = `from bayeswire import Data, Param, PartiallyObserved, model
from bayeswire.distributions import Normal

@model
class PartialTarget:
    theta = Param(Normal(0.0, 1.0))
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    y = PartiallyObserved.vector(
        Normal(theta, 1.0), length=n, observed=observed_values,
        observed_idx=observed_idx, missing_idx=missing_idx,
    )
`;
      const result = await compileScenario(source, `@model
class Shifted:
    theta = Param(Normal(2.0, 0.5))
`);
      assert(result.ok, `partial target composition failed: ${result.traceback}`);
      const document = JSON.parse(UTF8.decode(result.irBytes));
      assert(JSON.stringify(document).includes("VectorScatterOp"),
        "composed IR lost the target partially observed value");
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
