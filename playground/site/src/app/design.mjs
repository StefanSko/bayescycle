// Pure design-document helpers for the simulation workflow.

const DATA_FORMAT = "bayescycle.data.json.v1";

export function designDocument(design) {
  const variables = Object.fromEntries(
    Object.entries(design).map(([name, row]) => {
      if (Object.hasOwn(row, "value")) {
        if (!Number.isInteger(row.value) || row.value < 1) {
          throw new Error(`Design scalar ${name} must be a positive integer`);
        }
        return [name, { dtype: "int64", shape: [], values: [row.value] }];
      }
      const n = row.n ?? 50;
      if (
        !Number.isFinite(row.low) ||
        !Number.isFinite(row.high) ||
        !Number.isInteger(n) ||
        n < 1
      ) {
        throw new Error(
          `Design values for ${name} require finite low/high and a positive integer n`,
        );
      }
      const values = Array.from({ length: n }, (_, index) =>
        n === 1 ? row.low : row.low + ((row.high - row.low) * index) / (n - 1),
      );
      // Name-seeded permutations prevent shared linspaces from making predictors collinear.
      shuffle(values, mulberry32(fnv1a(name)));
      return [name, { dtype: "float64", shape: [n], values }];
    }),
  );
  return { format: DATA_FORMAT, variables };
}

export function truthDocument(truth) {
  const variables = Object.fromEntries(
    Object.entries(truth).map(([name, value]) => {
      if (!Number.isFinite(value)) {
        throw new Error(`Truth value for ${name} must be finite`);
      }
      return [name, { dtype: "float64", shape: [], values: [value] }];
    }),
  );
  return { format: DATA_FORMAT, variables };
}

export function defaultTruth(ir) {
  return Object.fromEntries(
    ir.model.params.map((parameter) => [
      parameter.name,
      parameter.value.constraint?.node === "Positive" ? 1 : 0,
    ]),
  );
}

export function designDefaults(ir) {
  const indexInputs = indexDataNames(ir.model);
  const vectors = ir.model.data.filter((input) => {
    const schema = input.value.schema;
    return schema.dims?.length === 1 || schema.rank === 1;
  });
  const derivedScalars = new Set(
    vectors.flatMap((input) => input.value.schema.dims?.map((dim) => dim.name) ?? []),
  );
  return Object.fromEntries(
    ir.model.data
      .filter((input) => {
        if (vectors.includes(input)) return true;
        const schema = input.value.schema;
        const scalar = (schema.dims?.length ?? 0) === 0 && !(schema.rank > 0);
        return scalar && !derivedScalars.has(input.name);
      })
      .map((input) => [
        input.name,
        vectors.includes(input)
          ? indexInputs.has(input.name)
            ? { low: 0, high: 0, n: 50 }
            : { low: -1, high: 1, n: 50 }
          : { value: 2 },
      ]),
  );
}

function indexDataNames(value, names = new Set()) {
  if (Array.isArray(value)) {
    for (const entry of value) indexDataNames(entry, names);
  } else if (value !== null && typeof value === "object") {
    if (value.node === "ScalarIndex" && value.expr?.node === "DataRef") {
      names.add(value.expr.name);
    }
    for (const entry of Object.values(value)) indexDataNames(entry, names);
  }
  return names;
}

function fnv1a(value) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x100000000;
  };
}

function shuffle(values, random) {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [values[index], values[swapIndex]] = [values[swapIndex], values[index]];
  }
}
