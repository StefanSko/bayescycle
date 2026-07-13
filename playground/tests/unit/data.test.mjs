import {
  bind,
  importCsv,
  importJson,
  parseCsv,
  requiredInputs,
  sniffDelimiter,
  standardize,
} from "/site/src/data/index.mjs";

const DIVORCE_FIXTURE = "/tests/fixtures/divorce/WaffleDivorce.csv";
const EIGHT_SCHOOLS_DATA = "/tests/fixtures/engine/eight_schools_non_centered/data.json";
const EIGHT_SCHOOLS_IR = "/tests/fixtures/corpus/eight_schools_non_centered.json";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertEqual(actual, expected, message) {
  const actualText = JSON.stringify(actual);
  const expectedText = JSON.stringify(expected);
  assert(actualText === expectedText, `${message}: expected ${expectedText}, got ${actualText}`);
}

function assertThrows(run, messagePart) {
  let error;
  try {
    run();
  } catch (caught) {
    error = caught;
  }
  assert(error !== undefined, "expected function to throw");
  assert(String(error).includes(messagePart), `expected error to name ${messagePart}: ${String(error)}`);
}

async function fetchText(path) {
  const response = await fetch(path);
  assert(response.ok, `fixture request for ${path} failed with HTTP ${response.status}`);
  return response.text();
}

function columnByName(result, name) {
  const column = result.columns.find((candidate) => candidate.name === name);
  assert(column !== undefined, `missing column ${name}`);
  return column;
}

export default [
  {
    name: "sniffs comma/semicolon/tab",
    fn: () => {
      for (const [delimiter, csv] of [
        [",", "a,b\n1,2\n"],
        [";", "a;b\n1;2\n"],
        ["\t", "a\tb\n1\t2\n"],
      ]) {
        assert(sniffDelimiter(csv) === delimiter, "delimiter sniffer disagrees");
        const result = importCsv(csv);
        assert(result.delimiter === delimiter, `expected ${JSON.stringify(delimiter)} delimiter`);
        assertEqual(result.columns.map((column) => column.name), ["a", "b"], "column names differ");
        assert(result.rowCount === 1, `expected one row, got ${result.rowCount}`);
        assertEqual(result.columns.map((column) => column.dtype), ["int64", "int64"], "column dtypes differ");
        assertEqual(result.columns.map((column) => column.values), [[1], [2]], "typed values differ");
      }
    },
  },
  {
    name: "falls back to comma on ties",
    fn: () => {
      const csv = "a,b;c\n1,2;3\n";
      assert(sniffDelimiter(csv) === ",", "delimiter sniffer did not fall back to comma");
      const result = importCsv(csv);
      assert(result.delimiter === ",", `expected comma fallback, got ${JSON.stringify(result.delimiter)}`);
    },
  },
  {
    name: "parses quoted fields with embedded delimiters",
    fn: () => {
      for (const [delimiter, embedded] of [
        [",", ";"],
        [";", ","],
        ["\t", ";"],
      ]) {
        const csv = `"given${embedded}name"${delimiter}note\r\n"Ada${embedded}Lovelace"${delimiter}"said ""hello""\r\nagain"\r\n`;
        assertEqual(
          parseCsv(csv, delimiter),
          [[`given${embedded}name`, "note"], [`Ada${embedded}Lovelace`, "said \"hello\"\r\nagain"]],
          "parsed quoted rows differ",
        );
        const result = importCsv(csv);
        assert(result.delimiter === delimiter, `expected ${JSON.stringify(delimiter)} delimiter`);
        assertEqual(result.columns.map((column) => column.name), ["given_name", "note"], "sanitized headers differ");
        assert(result.rowCount === 1, `expected one quoted row, got ${result.rowCount}`);
        assertEqual(
          result.columns.map((column) => column.values),
          [[`Ada${embedded}Lovelace`], ["said \"hello\"\r\nagain"]],
          "quoted values differ",
        );
        assertEqual(result.columns.map((column) => column.dtype), ["string", "string"], "quoted dtypes differ");
      }
      assertThrows(() => parseCsv('a,b\n"open,field', ","), "unterminated");
    },
  },
  {
    name: "single-column file does not warn; hidden delimiter does",
    fn: () => {
      const legitimate = importCsv("name\nAda\n\nGrace\n");
      assertEqual(legitimate.columns.map((column) => column.name), ["name"], "single-column header differs");
      assert(legitimate.rowCount === 2, `expected two non-blank rows, got ${legitimate.rowCount}`);
      assert(legitimate.suspectedDelimiter === undefined, "legitimate single column must not warn");

      const hidden = importCsv("name\nAda;Lovelace\n");
      assertEqual(hidden.columns.map((column) => column.name), ["name"], "hidden-delimiter header differs");
      assert(hidden.suspectedDelimiter === ";", `expected semicolon warning, got ${hidden.suspectedDelimiter}`);
    },
  },
  {
    name: "WaffleDivorce.csv: semicolon, 13 columns, 50 rows",
    fn: async () => {
      const result = importCsv(await fetchText(DIVORCE_FIXTURE));
      assert(result.delimiter === ";", `expected semicolon, got ${JSON.stringify(result.delimiter)}`);
      assert(result.columns.length === 13, `expected 13 columns, got ${result.columns.length}`);
      assert(result.rowCount === 50, `expected 50 rows, got ${result.rowCount}`);
      assert(result.suspectedDelimiter === undefined, "fixture must not warn about its delimiter");
      assert(columnByName(result, "Location").dtype === "string", "Location must be string");
      assert(columnByName(result, "MedianAgeMarriage").dtype === "float64", "MedianAgeMarriage must be float64");
    },
  },
  {
    name: "standardize gives mean 0 sd 1",
    fn: async () => {
      const imported = importCsv(await fetchText(DIVORCE_FIXTURE));
      const original = columnByName(imported, "MedianAgeMarriage");
      const before = JSON.stringify(original);
      const result = standardize(original);
      const mean = result.values.reduce((sum, value) => sum + value, 0) / result.values.length;
      const variance = result.values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / result.values.length;
      assert(Math.abs(mean) < 1e-9, `standardized mean is ${mean}`);
      assert(Math.abs(Math.sqrt(variance) - 1) < 1e-9, `standardized sd is ${Math.sqrt(variance)}`);
      assert(result.name === original.name, "standardization changed the column name");
      assert(result.dtype === "float64", `standardized dtype is ${result.dtype}`);
      assert(result.standardized === true, "standardized marker is absent");
      assert(JSON.stringify(original) === before, "standardization mutated its input");
      assertThrows(() => standardize(columnByName(imported, "Location")), "string");
      assertThrows(() => standardize({ name: "constant", dtype: "int64", values: [2, 2] }), "0");
    },
  },
  {
    name: "importJson validates data documents",
    fn: async () => {
      const text = await fetchText(EIGHT_SCHOOLS_DATA);
      assertEqual(importJson(text), JSON.parse(text), "valid data document did not round-trip");

      const mismatch = {
        format: "bayescycle.data.json.v1",
        variables: { x: { dtype: "float64", shape: [2], values: [1] } },
      };
      assertThrows(() => importJson(JSON.stringify(mismatch)), "x");

      const invalidEntry = {
        format: "bayescycle.data.json.v1",
        variables: { x: [1, 2] },
      };
      assertThrows(() => importJson(JSON.stringify(invalidEntry)), "x");
      assertThrows(() => importJson('{"format":"other","variables":{}}'), "format");
      assertThrows(() => importJson('{"format":"bayescycle.data.json.v1"}'), "variables");
    },
  },
  {
    name: "requiredInputs reads the IR",
    fn: async () => {
      const ir = JSON.parse(await fetchText(EIGHT_SCHOOLS_IR));
      assertEqual(
        requiredInputs(ir),
        [
          { name: "n_schools", kind: "scalar", dims: [] },
          { name: "sigma", kind: "vector", dims: ["n_schools"] },
          { name: "y", kind: "vector", dims: [] },
        ],
        "required model inputs differ",
      );
    },
  },
  {
    name: "bind matches by name, derives scalars, reports gaps",
    fn: async () => {
      const inputs = [
        { name: "n_schools", kind: "scalar", dims: [] },
        { name: "sigma", kind: "vector", dims: ["n_schools"] },
        { name: "y", kind: "vector", dims: [] },
      ];
      const columns = [
        { name: "sigma", dtype: "float64", values: [15, 10, 16, 11, 9, 11, 10, 18] },
        { name: "y", dtype: "float64", values: [28, 8, -3, 7, -1, 1, 18, 12] },
      ];
      const expected = JSON.parse(await fetchText(EIGHT_SCHOOLS_DATA));
      const result = bind(inputs, columns);
      assert(result.complete === true, "complete binding was reported incomplete");
      assertEqual(result.document, expected, "bound data document differs");
      assertEqual(
        result.mapping,
        [
          { input: "n_schools", source: "auto:length", status: "bound" },
          { input: "sigma", source: "column:sigma", status: "bound" },
          { input: "y", source: "column:y", status: "bound" },
        ],
        "complete mapping differs",
      );

      const incomplete = bind(inputs, columns.filter((column) => column.name !== "y"));
      assert(incomplete.complete === false, "binding without y was reported complete");
      assertEqual(
        incomplete.mapping.find((row) => row.input === "y"),
        { input: "y", source: null, status: "missing" },
        "missing y mapping differs",
      );
      assert(!("y" in incomplete.document.variables), "missing y leaked into variables");

      const incompatible = bind(inputs, [
        columns[0],
        { name: "y", dtype: "string", values: columns[1].values.map(String) },
      ]);
      assertEqual(
        incompatible.mapping.find((row) => row.input === "y"),
        { input: "y", source: null, status: "incompatible" },
        "incompatible y mapping differs",
      );
      assert(incompatible.complete === false, "incompatible binding was reported complete");
      assertThrows(
        () => bind(inputs, [{ ...columns[0], values: [null, ...columns[0].values.slice(1)] }, columns[1]]),
        "sigma",
      );
    },
  },
];
