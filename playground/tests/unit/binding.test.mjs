import { bind, importCsv } from "/site/src/data/index.mjs";

const DIVORCE_FIXTURE = "/tests/fixtures/divorce/WaffleDivorce.csv";
const INPUTS = [
  { name: "M", kind: "vector", dims: [] },
  { name: "A", kind: "vector", dims: [] },
  { name: "D", kind: "vector", dims: [] },
];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function mappingByInput(result, input) {
  return result.mapping.find((row) => row.input === input);
}

export default [
  {
    name: "bind assignments override name matching and report invalid columns",
    fn: async () => {
      const response = await fetch(DIVORCE_FIXTURE);
      assert(response.ok, `fixture request failed with HTTP ${response.status}`);
      const columns = importCsv(await response.text()).columns;
      const assignments = {
        M: "Marriage",
        A: "MedianAgeMarriage",
        D: "Divorce",
      };
      const bound = bind(INPUTS, columns, assignments);
      assert(bound.complete, "assigned divorce columns were not completely bound");
      for (const [input, column] of Object.entries(assignments)) {
        const mapping = mappingByInput(bound, input);
        assert(mapping.status === "bound", `${input} was not bound`);
        assert(mapping.source === `column:${column}`, `${input} source differs`);
      }

      const incompatible = bind(INPUTS, columns, { ...assignments, M: "Location" });
      assert(
        mappingByInput(incompatible, "M").status === "incompatible",
        "string assignment was not incompatible",
      );
      const missing = bind(INPUTS, columns, { ...assignments, M: "DoesNotExist" });
      assert(
        mappingByInput(missing, "M").status === "missing",
        "missing assignment was not reported missing",
      );
    },
  },
];
