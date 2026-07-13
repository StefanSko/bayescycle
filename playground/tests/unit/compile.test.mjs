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

export default [
  {
    name: "all 10 corpus models match native hashes and golden IR",
    fn: async () => {
      const hashes = JSON.parse(await fetchText("hashes.json"));
      const names = Object.keys(hashes);
      assert(names.length === 10, `expected 10 corpus models, found ${names.length}`);

      for (const name of names) {
        const [source, goldenText] = await Promise.all([
          fetchText(`${name}.py`),
          fetchText(`${name}.json`),
        ]);
        const result = await compile(source);
        assert(result.ok, `${name} compilation failed:\n${result.traceback}`);
        assert(
          result.irHash === hashes[name],
          `${name} hash mismatch: expected ${hashes[name]}, got ${result.irHash}`,
        );
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
    name: "compiler executes in a dedicated worker",
    fn: async () => {
      const source = await fetchText("linear_regression.py");
      const result = await compile(source);
      assert(result.ok, `worker compilation failed: ${result.traceback}`);
      assert(result.executionContext === "worker", `unexpected context: ${result.executionContext}`);
    },
  },
  {
    name: "user source cannot replace the trusted IR serializer",
    fn: async () => {
      const source = `import bayeswire.ir\nbayeswire.ir.canonical_bytes = lambda _meta: b"{}"\n${await fetchText("linear_regression.py")}`;
      const poisoned = await compile(source);
      assert(poisoned.ok, `poisoning fixture failed: ${poisoned.message}`);
      const hashes = JSON.parse(await fetchText("hashes.json"));
      assert(poisoned.irHash === hashes.linear_regression, `user serializer changed hash: ${poisoned.irHash}`);
      const clean = await compile(await fetchText("linear_regression.py"));
      assert(clean.ok && clean.irHash === hashes.linear_regression, "serializer mutation leaked into the next compile");
    },
  },
  {
    name: "trusted serializer is absent from user-visible globals",
    fn: async () => {
      const source = `import __main__\n__main__._trusted_canonical_bytes = lambda _meta: b"{}"\n${await fetchText("linear_regression.py")}`;
      const result = await compile(source);
      const hashes = JSON.parse(await fetchText("hashes.json"));
      assert(result.ok && result.irHash === hashes.linear_regression, `__main__ replaced trusted serializer: ${result.irHash}`);
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
